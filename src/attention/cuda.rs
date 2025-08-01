use crate::attention::{KVPage, KernelReq};
use cuda::{Module, Stream, params};
use std::ffi::c_uint;

pub trait NVDT: Copy {
    const COMPUTE_TYPE_NAME: &str;
    const DATA_TYPE_NAME: &str;
    const SIZE: usize;
}

impl NVDT for half::f16 {
    const COMPUTE_TYPE_NAME: &str = "float";
    const DATA_TYPE_NAME: &str = "half";
    const SIZE: usize = 2;
}

impl NVDT for half::bf16 {
    const COMPUTE_TYPE_NAME: &str = "float";
    const DATA_TYPE_NAME: &str = "nv_bfloat16";
    const SIZE: usize = 2;
}

impl NVDT for f32 {
    const COMPUTE_TYPE_NAME: &str = "float";
    const DATA_TYPE_NAME: &str = "float";
    const SIZE: usize = 4;
}

impl NVDT for f64 {
    const COMPUTE_TYPE_NAME: &str = "double";
    const DATA_TYPE_NAME: &str = "double";
    const SIZE: usize = 8;
}

impl super::FlashAttnCfg {
    pub fn compute_cuda<T: NVDT>(
        &self,
        cache_pages: &[KVPage<T>],
        reqs: &[KernelReq<T>],
        module: &Module,
        stream: &Stream,
    ) {
        let &Self {
            h, kvh, tile_seq, ..
        } = self;
        let n = reqs.len() as c_uint;
        let h = h as c_uint;
        let kvh = kvh as c_uint;
        let bn = tile_seq as c_uint;
        let warp = stream.ctx().dev().warp_size() as c_uint;
        // 拷贝参数
        let n_max = reqs.iter().map(|req| req.n).max().unwrap() as c_uint;
        let cache_pages = stream.from_host(cache_pages);
        let reqs_ = stream.from_host(reqs);
        // 发射 kernel
        let params = params![self.to_kernel_cfg(), cache_pages.as_ptr(), reqs_.as_ptr()];
        let nh_l = *[8, 4, 2, 1].iter().find(|&&n| kvh % n == 0).unwrap();
        stream
            .launch(
                &module.get_kernel(c"cache_concat"),
                ((n, kvh / nh_l), (nh_l, warp), 0),
                &params.to_ptrs(),
            )
            .launch(
                &module.get_kernel(c"flash_attn"),
                (
                    (n, h, n_max.div_ceil(bn)),
                    (bn, warp),
                    self.shared_elements() * size_of::<T>(),
                ),
                &params.to_ptrs(),
            )
            .free(reqs_)
            .free(cache_pages);
    }
}

pub fn code<T: NVDT>(d: usize, warp: usize) -> String {
    const CODE: &str = include_str!("cuda/kernel.cu");
    let t_data: &str = T::DATA_TYPE_NAME;
    let t_compute: &str = T::COMPUTE_TYPE_NAME;
    let ele_size = T::SIZE;
    let load = gen_load(d, ele_size, warp);
    format!(
        r#"
{load}
{CODE}

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
    )
}

#[cfg(test)]
impl super::FlashAttnCfg {
    pub(super) fn test_compute_cuda(
        &self,
        reqs: &mut [super::test::Attention],
        stream: &cuda::Stream,
    ) {
        use crate::attention::test::distinct;
        use any_tensor::digit_layout::types;

        let dt = reqs.iter().map(|req| req.dt()).collect::<Box<_>>();
        match distinct(&dt).unwrap() {
            types::F16 => self.test_compute_cuda_::<half::f16>(reqs, stream),
            types::BF16 => self.test_compute_cuda_::<half::bf16>(reqs, stream),
            types::F32 => self.test_compute_cuda_::<f32>(reqs, stream),
            types::F64 => self.test_compute_cuda_::<f64>(reqs, stream),
            others => panic!("Unsupported data type {others}"),
        }
    }

    fn test_compute_cuda_<T: NVDT>(
        &self,
        reqs: &mut [super::test::Attention],
        stream: &cuda::Stream,
    ) {
        use super::{KVPage, KernelReq, Strides2D};
        use cuda::Rtc;
        use std::iter::zip;

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
            .collect::<Box<_>>();

        let &Self { d, tile_ctx, .. } = self;
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
                    KVPage::<T> {
                        k: unsafe { base.byte_offset(k).cast_mut().cast() },
                        v: unsafe { base.byte_offset(v).cast_mut().cast() },
                    }
                })
            })
            .collect::<Box<_>>();
        // 为每个请求的每个头生成 block
        let reqs_ = reqs_o
            .iter()
            .scan(0, |start, (q, k, v, o, cache, pos)| {
                let pages_start = *start as _;
                let n = q.shape()[1];
                Some(KernelReq::<T> {
                    q: q.get().as_ptr().cast(),
                    q_strides: Strides2D::from_tensor(q),
                    k: k.get().as_ptr().cast(),
                    k_strides: Strides2D::from_tensor(k),
                    v: v.get().as_ptr().cast(),
                    v_strides: Strides2D::from_tensor(v),
                    pages_start,
                    kv_strides: Strides2D::from_tensor(
                        &cache
                            .as_ref()
                            .transform(|layout| layout.index(1, 0).transpose(&[1, 0])),
                    ),
                    o: o.get().as_ptr().cast_mut().cast(),
                    o_strides: Strides2D::from_tensor(o),
                    n,
                    s: n + pos,
                    ty: crate::attention::AttnType::Causal,
                    mask: std::ptr::null(),
                })
            })
            .collect::<Box<_>>();
        // 编译及计算
        let dev = stream.ctx().dev();
        let cc = dev.compute_capability();
        let warp = dev.warp_size();
        let program = Rtc::new().arch(cc).compile(&code::<T>(d, warp)).unwrap();
        let module = stream.ctx().load(&program);
        self.compute_cuda::<T>(&cache_pages, &reqs_, &module, stream);

        for ((.., o, cache, _), c) in zip(reqs_o, reqs) {
            stream
                .memcpy_d2h(c.cache.get_mut(), cache.get())
                .memcpy_d2h(c.o.get_mut(), o.get());
        }
    }
}

fn gen_load(d: usize, ele_size: usize, warp: usize) -> String {
    const CANDIDATES: &[[&str; 2]] = &[
        ["char", "0"],
        ["short", "0"],
        ["float", ".0f"],
        ["double", ".0"],
        ["float4", "float4{0,0,0,0}"],
        ["double4", "double4{0,0,0,0}"],
    ];

    let bytes = d * ele_size;
    let max = (bytes / warp).next_power_of_two().trailing_zeros() as usize;
    let (i, [t, zero]) = CANDIDATES
        .iter()
        .enumerate()
        .take(max + 1)
        .rev()
        .find(|(n, _)| bytes % (1 << n) == 0)
        .unwrap();
    let ele_size = 1 << i;
    let d = bytes / ele_size;

    const CODE: &str = include_str!("cuda/load.cu");
    format!(
        r#"{CODE}

__device__ __forceinline__ void load_qo(
    void *qi,
    void *oi,
    void const *q) {{
    load_qo_<{d}, {warp}>(({t} *)qi, ({t} *)oi, ({t} const*)q, {zero});
}}

__device__ __forceinline__ void load_kv(
    void *kj,
    void *vj,
    void const *k,
    void const *v) {{
    load_kv_<{d}, {warp}>(({t} *)kj, ({t} *)vj, ({t} const*)k, ({t} const*)v);
}}"#
    )
}
