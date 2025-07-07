use std::{
    ffi::{c_float, c_longlong, c_uint, c_ulonglong},
    iter::zip,
};

use any_tensor::digit_layout::types;
use cuda::{CurrentCtx, DevMem, Device, Ptx, memcpy_d2h};

use crate::attention::{Tensor, array, destruct, distinct};

use super::{Attention, FlashAttnCfg};

impl FlashAttnCfg {
    pub fn compute_cuda<'a>(&self, reqs: &mut [Attention]) {
        // 生成cuda环境
        assert!(cuda::init().is_ok());
        let device = Device::new(0);

        const CODE: &str = include_str!("kernel.cu");
        let (ptx, log) = Ptx::compile(CODE, device.compute_capability());
        let Ok(ptx) = ptx else { panic!("{log}") };

        device.context().apply(|ctx: &CurrentCtx| {
            let mut q_dev = Vec::with_capacity(reqs.len());
            let mut o_dev = Vec::with_capacity(reqs.len());
            let mut l_dev = Vec::with_capacity(reqs.len());
            let mut m_dev = Vec::with_capacity(reqs.len());
            let mut mask_dev = Vec::with_capacity(reqs.len());
            // TODO 之后应该在gpu上操作pages
            let mut pages_dev = Vec::with_capacity(reqs.len());
            let mut n_dev = Vec::with_capacity(reqs.len());
            let mut ts_dev = Vec::with_capacity(reqs.len());
            let mut kv_sbuf_dev = Vec::with_capacity(reqs.len());
            let mut kv_skv_dev = Vec::with_capacity(reqs.len());
            let mut kv_sh_dev = Vec::with_capacity(reqs.len());
            let len = reqs.len();
            // 向cuda复制内存，收集必要数据，
            for req in reqs.iter() {
                let Attention {
                    q,
                    k,
                    v,
                    o,
                    cache,
                    pos,
                } = req;
                let dt = distinct(&[q.dt(), k.dt(), v.dt(), o.dt(), cache.dt()]).unwrap();
                assert_eq!(dt, types::F64);
                // 解构形状维度
                destruct!([h_q, n_q, d_q] = q.shape());
                destruct!([h_o, n_o, d_o] = o.shape());
                destruct!([kvh_k, n_k, d_k] = k.shape());
                destruct!([kvh_v, n_v, d_v] = v.shape());
                destruct!([buf, 2, kvh_c, d_c] = cache.shape());
                let h = distinct(&[h_q, h_o]).unwrap();
                let n = distinct(&[n_q, n_k, n_v, n_o]).unwrap();
                let s = *pos + n;

                // 对齐张量形状
                q_dev.push(q.as_ref().map(|s| ctx.from_host(s)));
                o_dev.push(o.as_deref().map(|s| ctx.from_host(s)));
                l_dev.push(ctx.from_host(&vec![0.; h * s]));
                m_dev.push(ctx.from_host(&vec![f64::NEG_INFINITY; h * s]));
                // 生成 causal mask 这里也可以移动到gpu，让每个block共享mask
                let mask_data = (0..n * s)
                    .map(|i| i % s <= s - n + i / s)
                    .collect::<Vec<_>>();
                mask_dev.push(ctx.from_host(&mask_data));
                // pages 先使用cpu的page，之后移动到gpu
                destruct!([sbuf, skv, sh, sd] = cache.strides());
                assert_eq!(sd, dt.nbytes() as isize);
                assert_eq!(s % self.tile_ctx, 0);
                let pages = (0..s / self.tile_ctx)
                    .map(|i| {
                        let t = cache.as_ref().transform(|l| l.index(0, i * self.tile_ctx));
                        unsafe { t.get().as_ptr().byte_offset(t.offset()).cast_mut().cast() }
                    })
                    .collect::<Box<_>>();
                let pages = CachePages {
                    pages,
                    strides: [sbuf, skv, sh],
                    bs: self.tile_ctx,
                    dh: self.d,
                };
                // 连接 kv cache
                pages.concat(&k, &v, *pos);
                n_dev.push(n as c_ulonglong);
                ts_dev.push((s / self.tile_ctx) as c_ulonglong);
                pages_dev.push(pages.to_device(ctx));
                kv_sbuf_dev.push(sbuf as c_longlong);
                kv_skv_dev.push(skv as c_longlong);
                kv_sh_dev.push(sh as c_longlong);
            }
            let module = ctx.load(&ptx);
            let kernel = module.get_kernel(c"__attn_f64");
            let q_dev = ctx.from_host(&q_dev.iter().map(|s| s.get().as_ptr()).collect::<Vec<_>>());
            let o_dev_ = ctx.from_host(&o_dev.iter().map(|s| s.get().as_ptr()).collect::<Vec<_>>());
            let mask_dev = ctx.from_host(&mask_dev.iter().map(|s| s.as_ptr()).collect::<Vec<_>>());
            let m_dev = ctx.from_host(&m_dev.iter().map(|s| s.as_ptr()).collect::<Vec<_>>());
            let n_dev = ctx.from_host(&n_dev);
            let l_dev = ctx.from_host(&l_dev.iter().map(|s| s.as_ptr()).collect::<Vec<_>>());
            let pages_page = &pages_dev
                .iter()
                .map(|s| {
                    let data = s.iter().map(|d| d.as_ptr()).collect::<Vec<_>>();
                    ctx.from_host(&data)
                })
                .collect::<Vec<_>>();
            let pages_dev =
                ctx.from_host(&pages_page.iter().map(|f| f.as_ptr()).collect::<Vec<_>>());
            let ts_dev = ctx.from_host(&ts_dev);
            let kv_sbuf_dev = ctx.from_host(&kv_sbuf_dev);
            let kv_skv_dev = ctx.from_host(&kv_skv_dev);
            let kv_sh_dev = ctx.from_host(&kv_sh_dev);
            let params = cuda::params![
                pages_dev.as_ptr(),
                q_dev.as_ptr(),
                o_dev_.as_ptr(),
                mask_dev.as_ptr(),
                m_dev.as_ptr(),
                l_dev.as_ptr(),
                n_dev.as_ptr(),               // sequence length 转换成n指针
                self.d as c_ulonglong,        // head dim
                ts_dev.as_ptr(),              // = s/bs ts指针
                self.tile_ctx as c_ulonglong, // context tile
                (self.h / self.kvh) as c_ulonglong,
                self.d as c_longlong, //        q stride
                self.d as c_longlong, //        o stride
                kv_sbuf_dev.as_ptr(), // sequence stride 转换成指针
                kv_skv_dev.as_ptr(),  //   k to v stride
                kv_sh_dev.as_ptr(),   //  kv head stride
                (self.d as f64).sqrt().recip() as c_float
            ];
            for (i, p) in params.to_ptrs().iter().enumerate() {
                println!("param[{i}] = {:?}", p);
            }
            ctx.stream()
                .launch(
                    &kernel,
                    (
                        (len as c_uint, self.h as c_uint), // grid size
                        self.tile_ctx as c_uint,           // block size
                        (2 * self.tile_ctx * self.d) * std::mem::size_of::<f64>(),
                    ),
                    &*params.to_ptrs(),
                )
                .synchronize();
            // 从 device 拷贝结果
            // 批量复制复制
            for (req, o_data) in zip(reqs, o_dev) {
                let Attention { o, .. } = req;
                memcpy_d2h(o.get_mut(), o_data.get());
            }
        });
    }
}
struct CachePages {
    pages: Box<[*mut f64]>,
    /// \[buf, k|v, kvh]
    strides: [isize; 3],
    bs: usize,
    dh: usize,
}

impl CachePages {
    /// NOTICE GQA 需要发射此 kernel，MHA 可以合并到计算
    fn concat(&self, k: &Tensor<&[u8]>, v: &Tensor<&[u8]>, pos: usize) {
        destruct!([kvh_k, n_k, _d] = k.shape());
        destruct!([kvh_v, n_v, _d] = v.shape());

        let kvh = distinct(&[kvh_k, kvh_v]).unwrap();
        let n = distinct(&[n_k, n_v]).unwrap();
        for head in 0..kvh {
            let k = k.as_deref().transform(|l| l.index(0, head));
            let v = v.as_deref().transform(|l| l.index(0, head));

            for i in 0..n {
                let [k_cache, v_cache] = self.at(head, pos + i);
                k_cache.copy_from_slice(array(k.as_deref().transform(|l| l.index(0, i))));
                v_cache.copy_from_slice(array(v.as_deref().transform(|l| l.index(0, i))));
            }
        }
    }

    /// 定位指定 kv 头和位置的 token 切片
    fn at(&self, head: usize, pos: usize) -> [&mut [f64]; 2] {
        let [sbuf, skv, sh] = self.strides;
        let sh = head as isize * sh;
        let sbuf = (pos % self.bs) as isize * sbuf;
        let k = unsafe { self.pages[pos / self.bs].byte_offset(sh + sbuf) };
        let v = unsafe { k.byte_offset(skv) };
        unsafe { [k, v].map(|ptr| std::slice::from_raw_parts_mut(ptr, self.dh)) }
    }

    fn to_device<'a>(&self, ctx: &'a CurrentCtx) -> Vec<DevMem<'a>> {
        let [_sbuf, skv, _sh] = self.strides;
        // 1. 每一页数据拷贝到 device，收集 device pointer
        self.pages
            .iter()
            .map(|p| {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        *p as *const f64,
                        skv as usize * 2 * self.bs / std::mem::size_of::<f64>(),
                    )
                };
                ctx.from_host(data)
            })
            .collect::<Vec<_>>()
    }
}
