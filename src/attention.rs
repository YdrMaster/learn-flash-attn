use any_tensor::digit_layout::types;
use cuda::{CurrentCtx, DevMem, Device, Ptx, memcpy_d2h};
use std::{
    ffi::{c_float, c_int, c_long, c_longlong, c_uint, c_ulong, c_ulonglong},
    iter::zip,
};

macro_rules! destruct {
    ($pat:pat = $slice:expr) => {
        let &$pat = &*$slice else {
            panic!("Ndim mismatch ( = {})", $slice.len())
        };
    };
}

pub struct Attention<'a> {
    q: Tensor<&'a [u8]>,
    k: Tensor<&'a [u8]>,
    v: Tensor<&'a [u8]>,
    o: Tensor<&'a mut [u8]>,
    cache: Tensor<&'a mut [u8]>,
    pos: usize,
}

pub fn flash_attention(
    reqs: &mut [Attention],
    tile_seq: usize, // = tile q  = bn
    tile_ctx: usize, // = tile kv = bs
) {
    for req in reqs {
        let Attention {
            q,
            k,
            v,
            o,
            cache,
            pos,
        } = req;
        // 对齐数据类型
        let dt = distinct(&[q.dt(), k.dt(), v.dt(), o.dt(), cache.dt()]).unwrap();
        assert_eq!(dt, types::F64);
        // 解构形状维度
        destruct!([h_q, n_q, d_q] = q.shape());
        destruct!([h_o, n_o, d_o] = o.shape());
        destruct!([kvh_k, n_k, d_k] = k.shape());
        destruct!([kvh_v, n_v, d_v] = v.shape());
        destruct!([buf, 2, kvh_c, d_c] = cache.shape());
        // 对齐张量形状
        let h = distinct(&[h_q, h_o]).unwrap();
        let kvh = distinct(&[kvh_k, kvh_v, kvh_c]).unwrap();
        let n = distinct(&[n_q, n_k, n_v, n_o]).unwrap();
        let s = *pos + n;
        let d = distinct(&[d_q, d_k, d_v, d_o, d_c]).unwrap();
        assert!(buf >= s);
        // 计算标量参数
        assert_eq!(h % kvh, 0);
        let g = h / kvh;
        let scale = (d as f64).sqrt().recip();
        // kv 以页为单位分配
        assert_eq!(s % tile_ctx, 0);
        let pages = (0..s / tile_ctx)
            .map(|i| {
                let t = cache.as_ref().transform(|l| l.index(0, i * tile_ctx));
                unsafe { t.get().as_ptr().byte_offset(t.offset()).cast_mut().cast() }
            })
            .collect::<Box<_>>();
        destruct!([sbuf, skv, sh, sd] = cache.strides());
        assert_eq!(sd, dt.nbytes() as isize);
        let pages = CachePages {
            pages,
            strides: [sbuf, skv, sh],
            bs: tile_ctx,
        };
        // 生成 causal mask
        let mask_data = (0..n * s)
            .map(|i| i % s <= s - n + i / s)
            .collect::<Vec<_>>();
        let mask = Tensor::from_dim_slice(types::Bool, [n, s]).map(|_| erase_ty(&mask_data));

        // global
        let mut l = vec![0.; h * s]; // 注意力分母
        let mut m = vec![f64::NEG_INFINITY; h * s]; // 最大值缓存
        // 连接 kv cache
        pages.concat(&k, &v, *pos);
        // 计算注意力
        for head in 0..h {
            // local
            let mut qi = vec![0.; tile_seq * d]; // 暂存 q
            let mut kj = vec![0.; tile_ctx * d]; // 暂存 k
            let mut vj = vec![0.; tile_ctx * d]; // 暂存 v
            let mut x = vec![0.; tile_ctx * tile_seq]; // 注意力分数
            // block 之间完全无关，可以以任意方式并行
            FlashAttentionBlock {
                head: head / g,
                pages: &pages,

                q: &q.as_deref().transform(|l| l.index(0, head)),
                o: &mut o.as_deref_mut().transform(|l| l.index(0, head)),
                mask: &mask.as_deref(),
                l: &mut l[head * s..][..s],
                m: &mut m[head * s..][..s],
                qi: &mut qi,
                kj: &mut kj,
                vj: &mut vj,
                x: &mut x,
                n,
                s,
                d,
                bn: tile_seq,
                bs: tile_ctx,
                scale,
            }
            .launch()
        }
    }
}

struct CachePages {
    pages: Box<[*mut f64]>,
    strides: [isize; 3],
    bs: usize,
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
                let k = array::<f64>(k.as_deref().transform(|l| l.index(0, i)));
                let v = array::<f64>(v.as_deref().transform(|l| l.index(0, i)));
                unsafe { std::ptr::copy_nonoverlapping(k.as_ptr(), k_cache.cast(), k.len()) };
                unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), v_cache.cast(), v.len()) };
            }
        }
    }

    fn at(&self, head: usize, pos: usize) -> [*mut f64; 2] {
        let [sbuf, skv, sh] = self.strides;
        let sh = head as isize * sh;

        let sbuf = (pos % self.bs) as isize * sbuf;
        let base = unsafe { self.pages[pos / self.bs].byte_offset(sh + sbuf) };
        [base, unsafe { base.byte_offset(skv) }]
    }
    fn to_device<'a>(&'a self, ctx: &'a CurrentCtx) -> DevMem<'a> {
        let [_sbuf, skv, _sh] = self.strides;
        // 1. 每一页数据拷贝到 device，收集 device pointer
        let dev_pages: Vec<_> = self
            .pages
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
            .collect::<Vec<_>>();

        let dev_ptrs: Vec<_> = dev_pages.iter().map(|mem| mem.as_ptr()).collect();
        ctx.from_host(&dev_ptrs)
    }
}

struct FlashAttentionBlock<'a> {
    head: usize,
    pages: &'a CachePages,

    /// shape = {n, d}
    q: &'a Tensor<&'a [u8]>,
    /// shape = {n, d}
    o: &'a mut Tensor<&'a mut [u8]>,
    /// shape = {n, s}
    mask: &'a Tensor<&'a [u8]>,
    /// shape = {s}
    l: &'a mut [f64],
    /// shape = {s}
    m: &'a mut [f64],
    /// shape = {bn, d}
    qi: &'a mut [f64],
    /// shape = {bs, d}
    kj: &'a mut [f64],
    /// shape = {bs, d}
    vj: &'a mut [f64],
    /// shape = {bn, bs}
    x: &'a mut [f64],

    n: usize,
    s: usize,
    d: usize,
    bn: usize,
    bs: usize,
    scale: f64,
}

impl FlashAttentionBlock<'_> {
    fn launch(self) {
        let Self {
            head,
            pages,
            q,
            o,
            mask,
            l,
            m,
            qi,
            kj,
            vj,
            x,
            n,
            s,
            d,
            bn,
            bs,
            scale,
        } = self;

        for ikvb in 0..s.div_ceil(bs) {
            // thread
            for it in 0..bn {
                // 每个线程拷贝 k/v 的一行，拷贝整个 kv block 到 local memory
                let mut ikv = ikvb * bs + it;
                let mut i = it;
                while ikv < (ikvb + 1) * bs {
                    let [k, v] = pages.at(head, ikv);
                    ikv += bn;

                    unsafe { std::ptr::copy_nonoverlapping(k, kj[i * d..].as_mut_ptr(), d) };
                    unsafe { std::ptr::copy_nonoverlapping(v, vj[i * d..].as_mut_ptr(), d) };
                    i += bn;
                }
            }
            // thread
            for it in 0..bn {
                // 每个线程计算 q 的一行
                let qi = &mut qi[it * d..][..d];
                let x = &mut x[it * bs..][..bs];
                for iqb in 0..n.div_ceil(bn) {
                    let iq = iqb * bn + it;
                    if iq >= n {
                        break;
                    }
                    // locate data
                    let q = array::<f64>(q.as_deref().transform(|l| l.index(0, iq)));
                    let o = array_mut::<f64>(o.as_deref_mut().transform(|l| l.index(0, iq)));
                    let mask = array::<bool>(mask.as_deref().transform(|l| l.index(0, iq)));

                    // load data
                    qi.copy_from_slice(q);
                    let mask = &mask[ikvb * bs..][..bs];

                    let mi_1 = m[iq];
                    let di_1 = l[iq];

                    // score = q @ k^T / √d
                    let mut mi = mi_1;
                    for (i, (mask, x)) in zip(mask, &mut *x).enumerate() {
                        if !*mask {
                            *x = f64::NEG_INFINITY
                        } else {
                            let kj = &kj[i * d..][..d];
                            *x = zip(&*qi, kj).map(|(q, k)| q * k).sum::<f64>() * scale;
                            mi = f64::max(mi, *x)
                        }
                    }
                    let mut sum = 0.;
                    for x in &mut *x {
                        *x = (*x - mi).exp();
                        sum += *x
                    }

                    let mut exp = di_1 * (mi_1 - mi).exp();
                    let di = exp + sum;

                    m[iq] = mi;
                    l[iq] = di;

                    exp /= di;
                    x.iter_mut().for_each(|x| *x /= di);
                    let mut xv = vec![0.; o.len()];
                    for (i, x) in x.iter().enumerate() {
                        let vj = &vj[i * d..][..d];
                        zip(&mut xv, vj).for_each(|(xv, v)| *xv += x * v)
                    }
                    for (o, xv) in zip(o, xv) {
                        *o = *o * exp + xv
                    }
                }
            }
        }
    }
}

pub fn flash_attention_cuda(
    reqs: &mut [Attention],
    tile_seq: usize, // = tile q  = bn
    tile_ctx: usize, // = tile kv = bs
) {
    // 生成cuda环境
    assert!(cuda::init().is_ok());
    let device = Device::new(0);

    const CODE: &str = r#"
#include <cuda/std/cstdint>
#include <math_constants.h>

template <typename T>
struct kv_cache {
    T *k;
    T *v;
    __device__ kv_cache(T *k_, T *v_) : k(k_), v(v_) {}
};

template <typename T>
__device__ kv_cache<T> locate_cache(
    T *const *pages,
    int64_t sbuf,      // sequence stride
    int64_t skv,       //   k to v stride
    int64_t sh,        //  kv head stride
    uint64_t const bs, // context tile
    uint64_t const head,
    uint64_t const pos) {
    sh *= head;
    sbuf *= pos % bs;
    uint8_t *page = (uint8_t *)pages[pos / bs];
    return kv_cache{
        (T *)(page + sh + sbuf),
        (T *)(page + sh + sbuf + skv),
    };
}
template <typename T>
__device__ void __attn(
    T *const *kv_pages,
    T const *q_,           // n@ x d
    T *o_,                 // n@ x d
    bool const *mask_,     // n x s
    T *m,                  // s
    T *l,                  // s
    uint64_t const n,      // sequence length
    uint64_t const d,      // head dim
    uint64_t const ts,     // = s/bs
    uint64_t const bs,     // context tile
    uint64_t const g,      // GQA
    int64_t const sq,      //        q stride
    int64_t const so,      //        o stride
    int64_t const kv_sbuf, // sequence stride
    int64_t const kv_skv,  //   k to v stride
    int64_t const kv_sh,   //  kv head stride
    float const scale) {
    // (batch x head) x (bn)
    uint64_t const head = blockIdx.x;
    uint64_t const bn = blockDim.x;
    uint64_t const it = threadIdx.x;
    uint64_t const tn = (n + bn - 1) / bn;
    // 目前把所有维度当作连续的
    extern __shared__ T sram[];
    T *kj = sram;
    T *vj = sram + bs * d;
    const T *q = q_ + head * n * d;
    T *_o = o_ + head * n * d;
    for (uint64_t iqb = 0; iqb < tn; ++iqb) {
        uint64_t iq = iqb * bn + it;
        if (iq >= n) {
            continue;
        }

        T mi = m[iq];
        T li = l[iq];
        T *o = _o + iq * d;

        for (uint64_t ikvb = 0; ikvb < ts; ++ikvb) {
            // Load kv block to shared memory
            {
                uint64_t const end = (ikvb + 1) * bs;
                for (uint64_t ikv = ikvb * bs + it, i = it; ikv < end; ikv += bn, i += bn) {
                    kv_cache<T> cache = locate_cache<T>(kv_pages, kv_sbuf, kv_skv, kv_sh, bs, head/g, ikv);
                    for (uint64_t j = 0; j < d; ++j) {
                        kj[i * d + j] = cache.k[j];
                        vj[i * d + j] = cache.v[j];
                    }
                }
                __syncthreads();
            }

            // Load q_i and mask
            T* qi_val= new T[d];
            bool const *mask = mask_ + iq * bs * ts + ikvb * bs;

            for (uint64_t j = 0; j < d; ++j) {
                qi_val[j] = q[iq * d + j];
            }

            // Compute scores
             T* scores = new T[bs];
            T mi_local = -CUDART_INF_F;
            for (uint64_t i = 0; i < bs; ++i) {
                if (!mask[i]) {
                    scores[i] = -CUDART_INF_F;
                } else {
                    scores[i] = 0;
                    for (uint64_t j = 0; j < d; ++j) {
                        scores[i] += qi_val[j] * kj[i * d + j];
                    }
                    scores[i] *= scale;
                    if (scores[i] > mi_local) {
                        mi_local = scores[i];
                    }
                }
            }

            T mi_new = max(mi, mi_local);
            T sum = 0.0;

            for (uint64_t i = 0; i < bs; ++i) {
                if (mask[i]) {
                    scores[i] = exp(scores[i] - mi_new);
                    sum += scores[i];
                } else {
                    scores[i] = 0;
                }
            }

            T exp_old = (mi == -CUDART_INF_F) ? 0.0 : exp(mi - mi_new);
            T li_new = exp_old * li + sum;
            T rdi = (li_new == 0) ? 0.0 : 1.0 / li_new;

            // Update output
            for (uint64_t j = 0; j < d; ++j) {
                T v_acc = 0.0;
                for (uint64_t i = 0; i < bs; ++i) {
                    if (mask[i]) {
                        v_acc += scores[i] * vj[i * d + j];
                    }
                }
                o[j] = o[j] * exp_old * li * rdi + v_acc * rdi;
            }

            mi = mi_new;
            li = li_new;
        }

        m[iq] = mi;
        l[iq] = li;
    }
}

extern "C" __global__ void __attn_f64(
    double *const *kv_pages,
    double const *q_,
    double *o_,
    bool const *mask_,
    double *m,
    double *l,
    uint64_t const n,
    uint64_t const d,
    uint64_t const ts,
    uint64_t const bs,
    uint64_t const g,
    int64_t const sq,
    int64_t const so,
    int64_t const kv_sbuf,
    int64_t const kv_skv,
    int64_t const kv_sh,
    float const scale) {
    __attn<double>(
        kv_pages, q_, o_, mask_, m, l,
        n, d, ts, bs,g, sq, so,
        kv_sbuf, kv_skv, kv_sh, scale
    );
}
    "#;
    let (ptx, log) = Ptx::compile(CODE, device.compute_capability());

    let Ok(ptx) = ptx else { panic!("{log}") };
    for req in reqs {
        let Attention {
            q,
            k,
            v,
            o,
            cache,
            pos,
        } = req;
        // 对齐数据类型
        let dt = distinct(&[q.dt(), k.dt(), v.dt(), o.dt(), cache.dt()]).unwrap();
        assert_eq!(dt, types::F64);
        // 解构形状维度
        destruct!([h_q, n_q, d_q] = q.shape());
        destruct!([h_o, n_o, d_o] = o.shape());
        destruct!([kvh_k, n_k, d_k] = k.shape());
        destruct!([kvh_v, n_v, d_v] = v.shape());
        destruct!([buf, 2, kvh_c, d_c] = cache.shape());
        // 对齐张量形状
        let h = distinct(&[h_q, h_o]).unwrap();
        let kvh = distinct(&[kvh_k, kvh_v, kvh_c]).unwrap();
        let n = distinct(&[n_q, n_k, n_v, n_o]).unwrap();
        let s = *pos + n;
        let d = distinct(&[d_q, d_k, d_v, d_o, d_c]).unwrap();
        assert!(buf >= s);
        // 计算标量参数
        assert_eq!(h % kvh, 0);
        let g = h / kvh;
        let scale = (d as f64).sqrt().recip();
        // kv 以页为单位分配
        assert_eq!(s % tile_ctx, 0);
        let pages = (0..s / tile_ctx)
            .map(|i| {
                let t = cache.as_ref().transform(|l| l.index(0, i * tile_ctx));
                unsafe { t.get().as_ptr().byte_offset(t.offset()).cast_mut().cast() }
            })
            .collect::<Box<_>>();
        destruct!([sbuf, skv, sh, sd] = cache.strides());
        assert_eq!(sd, dt.nbytes() as isize);
        let pages = CachePages {
            pages,
            strides: [sbuf, skv, sh],
            bs: tile_ctx,
        };
        // 生成 causal mask
        let mask_data = (0..n * s)
            .map(|i| i % s <= s - n + i / s)
            .collect::<Vec<_>>();
        let [kv_sbuf, kv_skv, kv_sh] = pages.strides;
        // 连接 kv cache
        pages.concat(&k, &v, *pos);

        device.context().apply(|ctx| {
            let module = ctx.load(&ptx);
            let kernel = module.get_kernel(c"__attn_f64");

            let q = q.as_ref().map(|s| ctx.from_host(s));
            let o_ = o.as_deref().map(|s| ctx.from_host(s));
            let mask = ctx.from_host(&mask_data);

            let l = any_tensor::Tensor::new(types::F64, [h, s]).map(|len| ctx.malloc::<u8>(len));
            let m = ctx.from_host(&vec![f64::NEG_INFINITY; h * s]);
            //  TODO 必须放在最后，放在l,m之前会引起重复分配指针，导致pages和l,m指向同一内存
            let pages = pages.to_device(ctx);
            let params = cuda::params![
                pages.as_ptr(),
                q.get().as_ptr(),
                o_.get().as_ptr(),
                mask.as_ptr(),
                m.as_ptr(),
                l.get().as_ptr(),
                n as c_ulonglong,              // sequence length
                d as c_ulonglong,              // head dim
                (s / tile_ctx) as c_ulonglong, // = s/bs
                tile_ctx as c_ulonglong,       // context tile
                g as c_ulonglong,
                d_q as c_longlong,     //        q stride
                d_o as c_longlong,     //        o stride
                kv_sbuf as c_longlong, // sequence stride
                kv_skv as c_longlong,  //   k to v stride
                kv_sh as c_longlong,   //  kv head stride
                scale as c_float
            ];
            println!(
                "grid=({}, {}), block={}, shared_mem={} bytes",
                1,
                h,
                tile_ctx,
                (2 * tile_ctx * d) * std::mem::size_of::<f64>()
            );
            for (i, p) in params.to_ptrs().iter().enumerate() {
                println!("param[{i}] = {:?}", p);
            }
            ctx.stream()
                .launch(
                    &kernel,
                    (
                        (1 as c_uint, h as c_uint), // grid size
                        tile_ctx as c_uint,         // block size
                        (2 * tile_ctx * d) * std::mem::size_of::<f64>(),
                    ),
                    &*params.to_ptrs(),
                )
                .synchronize();
            // 从 device 拷贝结果
            memcpy_d2h(o.get_mut(), o_.get());
        });
    }
}

type Tensor<T> = any_tensor::Tensor<T, 3>;

fn distinct<T: Eq + Copy>(val: &[T]) -> Option<T> {
    let [ans, tail @ ..] = val else {
        return None;
    };
    if tail.iter().all(|x| x == ans) {
        Some(*ans)
    } else {
        None
    }
}

fn array<T: Copy>(tensor: Tensor<&[u8]>) -> &[T] {
    destruct!([n] = tensor.shape());
    destruct!([s] = tensor.strides());
    assert_eq!(tensor.dt().nbytes(), s as usize);

    let offset = tensor.offset() as usize;
    let data = &tensor.take()[offset..][..n * s as usize];
    let ([], data, []) = (unsafe { data.align_to::<T>() }) else {
        unreachable!()
    };
    data
}

fn array_mut<T: Copy>(tensor: Tensor<&mut [u8]>) -> &mut [T] {
    destruct!([n] = tensor.shape());
    destruct!([s] = tensor.strides());
    assert_eq!(tensor.dt().nbytes(), s as usize);

    let offset = tensor.offset() as usize;
    let data = &mut tensor.take()[offset..][..n * s as usize];
    let ([], data, []) = (unsafe { data.align_to_mut::<T>() }) else {
        unreachable!()
    };
    data
}

fn erase_ty<T: Copy>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::softmax::test::safe_softmax;

    #[test]
    fn test_flash_attention() {
        // 先计算不带cache的
        const H: usize = 8;
        const KVH: usize = 2;
        const N: usize = 4;
        const S: usize = 4;
        const P: usize = S - N;
        const D: usize = 20;

        let q = Tensor::from_dim_slice(types::F64, [H, N, D]);
        let o = Tensor::from_dim_slice(types::F64, [H, N, D]);
        let k = Tensor::from_dim_slice(types::F64, [KVH, N, D]);
        let v = Tensor::from_dim_slice(types::F64, [KVH, N, D]);
        let cache = Tensor::from_dim_slice(types::F64, [S, 2, KVH, D]);

        let q_data = random_data(H * N * D);
        let k_data = random_data(KVH * N * D);
        let v_data = random_data(KVH * N * D);

        let cache_data = random_data(S * 2 * KVH * D);

        let q = q.as_ref().map(|_| erase_ty(&q_data));
        let k = k.as_ref().map(|_| erase_ty(&k_data));
        let v = v.as_ref().map(|_| erase_ty(&v_data));
        let cache = cache.as_ref().map(|_| erase_ty(&v_data));

        // // 计算标准 attention
        let mut ans = vec![0.0f64; H * N * D];
        let mut cache_ans = cache_data.clone();
        attention(
            q.as_deref(),
            k.as_deref(),
            v.as_deref(),
            o.as_ref().map(|_| erase_ty_mut(&mut ans)),
            cache.as_ref().map(|_| erase_ty_mut(&mut cache_ans)),
            P,
        );

        // 计算 flash attention
        let mut res = vec![0.0f64; H * N * D];
        let mut cache_res = cache_data;
        let mut reqs = [Attention {
            q,
            k,
            v,
            o: o.as_ref().map(|_| erase_ty_mut(&mut res)),
            cache: cache.as_ref().map(|_| erase_ty_mut(&mut cache_res)),
            pos: P,
        }];
        flash_attention_cuda(&mut reqs, 2, 2);
        for (ans, res) in zip(ans, res).chain(zip(cache_ans, cache_res)) {
            let e_abs = (ans - res).abs();
            assert!(e_abs < 1e-7, "err = {e_abs:.3e} {}x", e_abs / f64::EPSILON)
        }
    }

    fn random_data(n: usize) -> Vec<f64> {
        (0..n)
            .map(|_| (rand::random::<f64>() - 0.5) * 10.)
            .collect()
    }

    fn erase_ty_mut<T: Copy>(data: &mut [T]) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), std::mem::size_of_val(data))
        }
    }

    /// 连接 cache
    fn cache_concat(
        k: &Tensor<&[u8]>,
        v: &Tensor<&[u8]>,
        cache: &mut Tensor<&mut [u8]>,
        pos: usize,
    ) {
        debug_assert!(matches!(
            distinct(&[k.dt(), v.dt(), cache.dt()]),
            Some(types::F64)
        ));

        destruct!([kvh_k, n_k, d_k] = k.shape());
        destruct!([kvh_v, n_v, d_v] = v.shape());
        destruct!([buf, 2, kvh_c, d_c] = cache.shape());

        let kvh = distinct(&[kvh_k, kvh_v, kvh_c]).unwrap();
        let n = distinct(&[n_k, n_v]).unwrap();
        let s = pos + n;
        debug_assert!(distinct(&[d_k, d_v, d_c]).is_some());
        debug_assert!(buf >= s);

        for i in 0..kvh {
            let k = k.as_deref().transform(|l| l.index(0, i));
            let v = v.as_deref().transform(|l| l.index(0, i));
            let mut cache = cache.as_deref_mut().transform(|l| l.index(2, i));

            for i in 0..n {
                let mut cache = cache.as_deref_mut().transform(|l| l.index(0, pos + i));
                let k = array::<f64>(k.as_deref().transform(|l| l.index(0, i)));
                let v = array::<f64>(v.as_deref().transform(|l| l.index(0, i)));
                array_mut::<f64>(cache.as_deref_mut().transform(|l| l.index(0, 0)))
                    .copy_from_slice(k);
                array_mut::<f64>(cache.as_deref_mut().transform(|l| l.index(0, 1)))
                    .copy_from_slice(v);
            }
        }
    }

    /// 多头注意力计算
    pub fn attention(
        q: Tensor<&[u8]>,
        k: Tensor<&[u8]>,
        v: Tensor<&[u8]>,
        mut o: Tensor<&mut [u8]>,
        mut cache: Tensor<&mut [u8]>,
        pos: usize,
    ) {
        // 对齐数据类型
        let dt = distinct(&[q.dt(), k.dt(), v.dt(), o.dt(), cache.dt()]).unwrap();
        assert_eq!(dt, types::F64);
        // 解构形状维度
        destruct!([h_q, n_q, d_q] = q.shape());
        destruct!([h_o, n_o, d_o] = o.shape());
        destruct!([kvh_k, n_k, d_k] = k.shape());
        destruct!([kvh_v, n_v, d_v] = v.shape());
        destruct!([buf, 2, kvh_c, d_c] = cache.shape());
        // 对齐张量形状
        let h = distinct(&[h_q, h_o]).unwrap();
        let kvh = distinct(&[kvh_k, kvh_v, kvh_c]).unwrap();
        let n = distinct(&[n_q, n_k, n_v, n_o]).unwrap();
        let s = pos + n;
        let d = distinct(&[d_q, d_k, d_v, d_o, d_c]).unwrap();
        assert!(buf >= s);
        // 计算标量参数
        assert_eq!(h % kvh, 0);
        let g = h / kvh;
        let scale = (d as f64).sqrt().recip();
        // 连接 kv cache
        cache_concat(&k, &v, &mut cache, pos);
        // 计算注意力
        let k = cache.as_deref().transform(|l| l.index(1, 0));
        let v = cache.as_deref().transform(|l| l.index(1, 1));
        for i in 0..h {
            let q = q.as_deref().transform(|l| l.index(0, i));
            let k = k.as_deref().transform(|l| l.index(1, i / g));
            let v = v.as_deref().transform(|l| l.index(1, i / g));
            let mut o = o.as_deref_mut().transform(|l| l.index(0, i));

            for i in 0..n {
                let mut score = vec![0.; s];
                let q = array::<f64>(q.as_deref().transform(|l| l.index(0, i)));
                let o = array_mut::<f64>(o.as_deref_mut().transform(|l| l.index(0, i)));
                // score = q @ k^T / √d
                let len = s - n + i + 1;
                for (i, score) in score.iter_mut().enumerate().take(len) {
                    let k = array::<f64>(k.as_deref().transform(|l| l.index(0, i)));
                    *score = zip(q, k).map(|(q, k)| q * k).sum::<f64>() * scale
                }
                // causal softmax
                safe_softmax(&mut score[..len]);
                // o = a @ v // 乘法不连续
                o.fill(0.);
                for (i, score) in score.iter().enumerate().take(len) {
                    let v = array::<f64>(v.as_deref().transform(|l| l.index(0, i)));
                    for (v, o) in zip(v, &mut *o) {
                        *o += score * v
                    }
                }
            }
        }
    }
}
