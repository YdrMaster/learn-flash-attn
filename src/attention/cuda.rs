use super::{Attention, destruct};
use crate::attention::{
    FlashAttnCfg, cache_concat,
    kernel::{KVPage, KernelReq, Strides2D},
};
use cuda::{Ptx, Stream, params};
use std::ffi::c_uint;

impl FlashAttnCfg {
    pub fn compute_cuda(&self, reqs: &mut [Attention], stream: &Stream) {
        // 连接 kv cache
        for req in &mut *reqs {
            cache_concat(&req.k, &req.v, &mut req.cache, req.pos)
        }
        // 生成 GPU 版本
        let reqs_ = reqs
            .iter()
            .map(|req| {
                (
                    req.q.as_ref().map(|s| stream.from_host(s)),
                    req.o.as_ref().map(|s| stream.from_host(s)),
                    req.cache.as_ref().map(|s| stream.from_host(s)),
                    req.pos,
                )
            })
            .collect::<Vec<_>>();
        // 预备参数
        let &Self {
            h,
            tile_seq,
            tile_ctx,
            ..
        } = self;
        // 生成发送给 kernel 的配置
        let cfg = self.to_kernel_cfg();
        // 生成所有页指针
        let cache_pages = reqs_
            .iter()
            .flat_map(|(q, _, cache, pos)| {
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
        let req_memory = reqs_
            .iter()
            .map(|(q, _, _, pos)| {
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
        let reqs_ = reqs_
            .iter()
            .zip(&req_memory)
            .scan(0, |start, ((q, o, cache, pos), mem)| {
                let pages_start = *start as _;
                let n = q.shape()[1];
                Some(KernelReq {
                    q: q.get().as_ptr().cast(),
                    q_strides: {
                        destruct!([head, seq, _] = q.strides());
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
        const CODE: &str = include_str!("kernel.cu");
        let cc = stream.ctx().dev().compute_capability();
        let (ptx, log) = Ptx::compile(CODE, cc);
        let ptx = match ptx {
            Ok(ptx) => {
                if !log.is_empty() {
                    println!("{log}")
                }
                ptx
            }
            Err(e) => panic!("{e:?}\n{log}"),
        };
        let module = stream.ctx().load(&ptx);
        let kernel = module.get_kernel(c"flash_attn");

        stream.launch(
            &kernel,
            (
                (reqs.len() as c_uint, h as c_uint),
                tile_seq as c_uint,
                self.shared_elements() * size_of::<f64>(),
            ),
            &params![cfg, cache_pages.as_ptr(), reqs_.as_ptr()].to_ptrs(),
        );

        todo!()
    }
}
