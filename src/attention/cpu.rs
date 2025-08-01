use super::{AttnType, KVPage, KernelCfg, KernelReq};
use num_traits::{Float, FromPrimitive};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    iter::{Sum, zip},
    ops::AddAssign,
    slice::{from_raw_parts, from_raw_parts_mut},
};

impl super::FlashAttnCfg {
    pub fn compute_cpu<T>(&self, cache_pages: &[KVPage<T>], reqs: &[KernelReq<T>])
    where
        T: Float + FromPrimitive + AddAssign + Sum<T>,
    {
        let &Self {
            h, kvh, tile_seq, ..
        } = self;
        // 发射 kernel
        let cfg = self.to_kernel_cfg();
        launch_parallel(1, reqs.len(), kvh, |[_, ireq, head]| {
            cache_concat_block(cfg, cache_pages, reqs, ireq, head)
        });
        launch_parallel(reqs.len(), h, tile_seq, |indices| {
            let mut shared = vec![T::zero(); self.shared_elements()];
            flash_attn_block(cfg, cache_pages, reqs, indices, tile_seq, &mut shared)
        });
    }
}

fn launch_parallel(z: usize, y: usize, x: usize, f: impl Fn([usize; 3]) + Sync) {
    (0..z)
        .into_par_iter()
        .flat_map(|z| (0..y).into_par_iter().map(move |y| [z, y]))
        .flat_map(|[z, y]| (0..x).into_par_iter().map(move |x| [z, y, x]))
        .for_each(&f)
}

fn cache_concat_block<T: Copy>(
    cfg: KernelCfg,
    cache_pages: &[KVPage<T>],
    reqs: &[KernelReq<T>],
    ireq: usize,
    head: usize,
) {
    let KernelCfg { d, bs, .. } = cfg;
    let &KernelReq {
        k,
        k_strides,
        v,
        v_strides,
        pages_start,
        kv_strides,
        n,
        s,
        ..
    } = &reqs[ireq];
    let pages = &cache_pages[pages_start..];
    let pos = s - n;
    for i in 0..n {
        let page = pages[(pos + i) / bs];
        let k_offset = k_strides.offset(head, i);
        let v_offset = v_strides.offset(head, i);
        let c_offset = kv_strides.offset(head, (pos + i) % bs);
        for it in 0..d {
            unsafe {
                *page.k.byte_offset(c_offset).add(it) = *k.byte_offset(k_offset).add(it);
                *page.v.byte_offset(c_offset).add(it) = *v.byte_offset(v_offset).add(it);
            }
        }
    }
}

fn flash_attn_block<T: Float + FromPrimitive + AddAssign + Sum<T>>(
    cfg: KernelCfg,
    cache_pages: &[KVPage<T>],
    reqs: &[KernelReq<T>],
    indices: [usize; 3],
    bn: usize,
    shared: &mut [T],
) {
    let [ireq, head, q_base] = indices;
    let KernelCfg { g, d, bs, scale } = cfg;
    let scale = T::from_f32(scale).unwrap();
    let &KernelReq {
        q,
        q_strides,
        pages_start,
        kv_strides,
        o,
        o_strides,
        n,
        s,
        ty,
        mask,
        ..
    } = &reqs[ireq];
    let pages = &cache_pages[pages_start..][..s.div_ceil(bs)];
    // 划分 shared memory
    let (qi, shared) = shared.split_at_mut(d);
    let (oi, shared) = shared.split_at_mut(d);
    let (kj, shared) = shared.split_at_mut(bs * d);
    let (vj, x) = shared.split_at_mut(bs * d);
    assert_eq!(x.len(), bs);
    // seq 方向迭代
    for iq in (q_base..n).step_by(bn) {
        let q = unsafe { from_raw_parts(q.byte_offset(q_strides.offset(head, iq)), d) };
        let o = unsafe { from_raw_parts_mut(o.byte_offset(o_strides.offset(head, iq)), d) };
        // load q/o
        qi.copy_from_slice(q);
        oi.fill(T::zero());
        // 初始化 m l
        let mut mi_1 = T::neg_infinity();
        let mut di_1 = T::zero();
        // ctx 方向迭代
        for (ikvb, KVPage { k, v }) in pages.iter().enumerate() {
            // 每个线程拷贝 k/v 的一行，拷贝整个 kv block 到 local memory
            for i in 0..bs {
                let offset = kv_strides.offset(head / g, i);
                kj[i * d..][..d]
                    .copy_from_slice(unsafe { from_raw_parts(k.byte_offset(offset), d) });
                vj[i * d..][..d]
                    .copy_from_slice(unsafe { from_raw_parts(v.byte_offset(offset), d) });
            }
            // 计算当前线程 qi oi x 对应的索引
            let mask = unsafe { from_raw_parts(mask.add((iq * pages.len() + ikvb) * bs), bs) };

            // 用于计算 mask
            let pos = s - n;
            let kv_pos_base = ikvb * bs;

            // score = q @ k^T / √d
            let mut mi = mi_1;
            for (i, (mask, x)) in zip(mask, &mut *x).enumerate() {
                let is_valid = match ty {
                    AttnType::Full => true,
                    AttnType::Causal => kv_pos_base + i <= pos + iq,
                    AttnType::CustomMask => *mask,
                };

                if is_valid {
                    let qk = zip(&*qi, &kj[i * d..][..d])
                        .map(|(&q, &k)| q * k)
                        .sum::<T>()
                        * scale;
                    *x = qk;
                    if qk > mi {
                        mi = qk
                    }
                }
            }

            let mut sum = T::zero();
            for (i, (mask, x)) in zip(mask, &mut *x).enumerate() {
                let is_valid = match ty {
                    AttnType::Full => true,
                    AttnType::Causal => kv_pos_base + i <= pos + iq,
                    AttnType::CustomMask => *mask,
                };

                if !is_valid {
                    *x = T::zero()
                } else {
                    *x = (*x - mi).exp();
                    sum += *x
                }
            }

            let exp = (mi_1 - mi).exp();
            let di = di_1 * exp + sum;
            // 更新 m, l
            mi_1 = mi;
            di_1 = di;
            for (j, oi) in oi.iter_mut().enumerate() {
                let xv = x
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| match ty {
                        AttnType::Full => true,
                        AttnType::Causal => kv_pos_base + i <= pos + iq,
                        AttnType::CustomMask => mask[i],
                    })
                    .map(|(i, x)| *x * vj[i * d + j])
                    .sum::<T>();
                *oi = *oi * exp + xv
            }
        }
        // 将 oi 写入 o
        let k = di_1.recip();
        for (o, oi) in zip(o, &mut *oi) {
            *o = *oi * k
        }
    }
}

#[cfg(test)]
impl super::FlashAttnCfg {
    pub(super) fn test_compute_cpu(&self, reqs: &mut [super::test::Attention]) {
        use crate::attention::test::distinct;
        use any_tensor::digit_layout::types;

        let dt = reqs.iter().map(|req| req.dt()).collect::<Box<_>>();
        match distinct(&dt).unwrap() {
            types::F16 => self.test_compute_cpu_::<half::f16>(reqs),
            types::BF16 => self.test_compute_cpu_::<half::bf16>(reqs),
            types::F32 => self.test_compute_cpu_::<f32>(reqs),
            types::F64 => self.test_compute_cpu_::<f64>(reqs),
            others => panic!("Unsupported data type {others}"),
        }
    }

    fn test_compute_cpu_<T: Float + FromPrimitive + AddAssign + Sum<T>>(
        &self,
        reqs: &mut [super::test::Attention],
    ) {
        use super::{AttnType, Strides2D, test::Attention};

        let &Self { tile_ctx, .. } = self;
        // 生成所有页指针
        let cache_pages = reqs
            .iter()
            .flat_map(|req: &Attention<'_>| {
                let n = req.q.shape()[1];
                let pos = req.pos;
                (0..(pos + n).div_ceil(tile_ctx)).map(|i| {
                    let cache = req
                        .cache
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
                        k: unsafe { base.byte_offset(k).cast_mut().cast() },
                        v: unsafe { base.byte_offset(v).cast_mut().cast() },
                    }
                })
            })
            .collect::<Box<_>>();
        // 生成 mask
        let masks = reqs
            .iter()
            .map(|req| {
                let n = req.q.shape()[1];
                let pos = req.pos;
                let s = pos + n;
                let s_ceil = s.div_ceil(tile_ctx) * tile_ctx;
                // 注意力掩码
                (0..n * s_ceil)
                    .map(|i| i % s_ceil <= s - n + i / s_ceil)
                    .collect::<Box<_>>()
            })
            .collect::<Box<_>>();
        // 为每个请求的每个头生成 block
        let reqs = reqs
            .iter()
            .zip(&masks)
            .scan(0, |start, (req, mem)| {
                let pages_start = *start as _;
                let n = req.q.shape()[1];
                Some(KernelReq {
                    q: req.q.get().as_ptr().cast(),
                    q_strides: Strides2D::from_tensor(&req.q),
                    k: req.k.get().as_ptr().cast(),
                    k_strides: Strides2D::from_tensor(&req.k),
                    v: req.v.get().as_ptr().cast(),
                    v_strides: Strides2D::from_tensor(&req.v),
                    pages_start,
                    kv_strides: Strides2D::from_tensor(
                        &req.cache
                            .as_ref()
                            .transform(|layout| layout.index(1, 0).transpose(&[1, 0])),
                    ),
                    o: req.o.get().as_ptr().cast_mut().cast(),
                    o_strides: Strides2D::from_tensor(&req.o),
                    mask: mem.as_ptr(),
                    n,
                    s: req.pos + n,
                    ty: AttnType::Causal,
                })
            })
            .collect::<Box<_>>();

        self.compute_cpu::<T>(&cache_pages, &reqs)
    }
}
