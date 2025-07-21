use super::{Attention, destruct};
use crate::attention::{
    FlashAttnCfg,
    kernel::{KVPage, KernelReq, Strides2D},
};
use cuda::{CurrentCtx, Module, Ptx, Stream, params};
use std::{ffi::c_uint, iter::zip};

impl FlashAttnCfg {
    pub fn compute_cuda(&self, reqs: &mut [Attention], stream: &Stream) {
        // 生成 GPU 版本
        let reqs_o = reqs
            .iter()
            .map(|req| {
                (
                    req.q.as_ref().map(|s| stream.from_host(s)),
                    req.k.as_ref().map(|s| stream.from_host(s)),
                    req.v.as_ref().map(|s| stream.from_host(s)),
                    req.o.as_ref().map(|s| stream.from_host(s)),
                    req.cache.as_ref().map(|s| stream.from_host(s)),
                    req.pos,
                )
            })
            .collect::<Vec<_>>();
        // 预备参数
        let &Self {
            h,
            kvh,
            d,
            tile_seq,
            tile_ctx,
        } = self;
        // 生成发送给 kernel 的配置
        let cfg = self.to_kernel_cfg();
        // 生成所有页指针
        let cache_pages = reqs_o
            .iter()
            .flat_map(|(q, _, _, _, cache, pos)| {
                let n = q.shape()[1];
                (0..(pos + n).div_ceil(tile_ctx)).map(|i| {
                    let cache = cache
                        .as_ref()
                        .transform(|layout| layout.index(0, i * tile_ctx));
                    let base = cache.get().as_ptr();
                    let k = cache
                        .as_ref()
                        .transform(|layout| layout.index(0, 0))
                        .offset();
                    let v = cache
                        .as_ref()
                        .transform(|layout| layout.index(0, 1))
                        .offset();
                    KVPage {
                        k: unsafe { base.byte_offset(k).cast() },
                        v: unsafe { base.byte_offset(v).cast() },
                    }
                })
            })
            .collect::<Vec<_>>();
        // 生成 workspace
        let req_memory = reqs_o
            .iter()
            .map(|(q, _, _, _, _, pos)| {
                let n = q.shape()[1];
                let s = pos + n;
                // 注意力掩码
                let mask = (0..n * s)
                    .map(|i| i % s <= s - n + i / s)
                    .collect::<Box<_>>();
                // 注意力分母
                let l = vec![0.; h * s];
                // 最大值缓存
                let m = vec![f64::NEG_INFINITY; h * s];
                (
                    stream.from_host(&mask),
                    stream.from_host(&l),
                    stream.from_host(&m),
                )
            })
            .collect::<Vec<_>>();
        // 为每个请求的每个头生成 block
        let reqs_ = reqs_o
            .iter()
            .zip(&req_memory)
            .scan(0, |start, ((q, k, v, o, cache, pos), mem)| {
                let pages_start = *start as _;
                let n = q.shape()[1];
                Some(KernelReq {
                    q: q.get().as_ptr().cast(),
                    q_strides: {
                        destruct!([head, seq, _] = q.strides());
                        Strides2D { head, seq }
                    },

                    k: k.get().as_ptr().cast(),
                    k_strides: {
                        destruct!([head, seq, _] = k.strides());
                        Strides2D { head, seq }
                    },
                    v: v.get().as_ptr().cast(),
                    v_strides: {
                        destruct!([head, seq, _] = v.strides());
                        Strides2D { head, seq }
                    },
                    pages_start,
                    kv_strides: {
                        destruct!([seq, _, head, _] = cache.strides());
                        Strides2D { head, seq }
                    },
                    o: o.get().as_ptr().cast_mut().cast(),
                    o_strides: {
                        destruct!([head, seq, _] = o.strides());
                        Strides2D { head, seq }
                    },
                    mask: mem.0.as_ptr().cast(),
                    l: mem.1.as_ptr().cast_mut().cast(),
                    m: mem.2.as_ptr().cast_mut().cast(),
                    n,
                    s: n + pos,
                })
            })
            .collect::<Vec<_>>();
        let cache_pages = stream.from_host(&cache_pages);
        let reqs_ = stream.from_host(&reqs_);
        // 编译
        let module = compile("double", "double", stream.ctx());
        let params = params![cfg, cache_pages.as_ptr(), reqs_.as_ptr()];

        stream
            .launch(
                &module.get_kernel(c"cache_concat"),
                (reqs.len() as c_uint, (kvh as c_uint, d as c_uint), 0),
                &params.to_ptrs(),
            )
            .launch(
                &module.get_kernel(c"flash_attn"),
                (
                    (reqs.len() as c_uint, h as c_uint),
                    tile_seq as c_uint,
                    self.shared_elements() * size_of::<f64>(),
                ),
                &params.to_ptrs(),
            );

        for ((.., o, cache, _), c) in zip(reqs_o, reqs) {
            stream
                .memcpy_d2h(c.cache.get_mut(), cache.get())
                .memcpy_d2h(c.o.get_mut(), o.get());
        }
    }
}

fn compile<'ctx>(t_compute: &str, t_data: &str, ctx: &'ctx CurrentCtx) -> Module<'ctx> {
    const CODE: &str = include_str!("kernel.cuh");
    let code = format!(
        r#"{CODE}

extern "C" __global__ void cache_concat(
    KernelCfg cfg,
    KVPage<{t_data}> const *cache_pages,
    KernelReq<{t_data}> const *reqs) {{
    cache_concat_block(cfg, cache_pages, reqs);
}}

extern "C" __global__ void flash_attn(
    KernelCfg cfg,
    KVPage<{t_data}> const *cache_pages,
    KernelReq<{t_data}> const *reqs) {{
    flash_attn_block<{t_compute}>(cfg, cache_pages, reqs);
}}"#
    );
    let cc = ctx.dev().compute_capability();
    let (ptx, log) = Ptx::compile(code, cc);
    let ptx = match ptx {
        Ok(ptx) => {
            if !log.is_empty() {
                println!("{log}")
            }
            ptx
        }
        Err(e) => panic!("{e:?}\n{log}"),
    };
    ctx.load(&ptx)
}
