use crate::attention::{KVPage, KernelCfg, KernelReq};
use num_traits::{Float, FromPrimitive};
use std::{
    iter::{Sum, zip},
    ops::AddAssign,
    ptr::copy_nonoverlapping,
    slice::{from_raw_parts, from_raw_parts_mut},
};

pub fn cache_concat_block<T: Copy>(
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

pub fn flash_attn_block<T: Float + FromPrimitive + AddAssign + Sum<T>>(
    cfg: KernelCfg,
    cache_pages: &[KVPage<T>],
    reqs: &[KernelReq<T>],
    ireq: usize,
    head: usize,
    bn: usize,
    shared: &mut [T],
) {
    let KernelCfg { g, d, bs, scale } = cfg;
    let scale = T::from_f32(scale).unwrap();
    let &KernelReq {
        q,
        q_strides,
        pages_start,
        kv_strides,
        o,
        o_strides,
        mask,
        n,
        s,
        ..
    } = &reqs[ireq];
    let pages = &cache_pages[pages_start..][..s.div_ceil(bs)];
    // 划分 shared memory
    let (qi, shared) = shared.split_at_mut(bn * d);
    let (oi_, shared) = shared.split_at_mut(bn * d);
    let (kj, shared) = shared.split_at_mut(bs * d);
    let (vj, x) = shared.split_at_mut(bs * d);
    assert_eq!(x.len(), bn * bs);
    // seq 方向迭代
    for iqb in 0..n.div_ceil(bn) {
        // 每个线程拷贝 q 的一行，拷贝整个 q block 到 local memory
        for it in 0..bn {
            let offset = q_strides.offset(head, iqb * bn + it);
            unsafe {
                copy_nonoverlapping(q.byte_offset(offset), qi[it * d..].as_mut_ptr(), d);
            }
        }

        //初始化oi为0
        oi_.fill(T::zero());

        //不必从外面传进来，因为每个线程可以很方便地持有一个
        let mut mi_1_array = vec![T::neg_infinity(); bn];
        let mut di_1_array = vec![T::zero(); bn];

        // ctx 方向迭代
        for (ikvb, KVPage { k, v }) in pages.iter().enumerate() {
            // 每个线程拷贝 k/v 的一行，拷贝整个 kv block 到 local memory
            for it in 0..bn {
                for i in (it..bs).step_by(bn) {
                    let offset = kv_strides.offset(head / g, i);
                    unsafe {
                        copy_nonoverlapping(k.byte_offset(offset), kj[i * d..].as_mut_ptr(), d);
                        copy_nonoverlapping(v.byte_offset(offset), vj[i * d..].as_mut_ptr(), d);
                    }
                }
            }

            // 每个线程计算 q 的一行
            for it in 0..bn {
                // 计算当前线程qi oi x 对应的索引
                let qi = &mut qi[it * d..][..d];
                let oi_ = &mut oi_[it * d..][..d];
                let x = &mut x[it * bs..][..bs];

                let iq = iqb * bn + it;
                if iq >= n {
                    break;
                }

                let mask = unsafe { from_raw_parts(mask.add(iq * s + ikvb * bs), bs) };

                // 初始化 mi_1, di_1
                let mi_1 = &mut mi_1_array[it];
                let di_1 = &mut di_1_array[it];

                // score = q @ k^T / √d
                let mut mi = *mi_1;
                for (i, (mask, x)) in zip(mask, &mut *x).enumerate() {
                    if !*mask {
                        *x = T::neg_infinity();
                    } else {
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
                for x in &mut *x {
                    *x = (*x - mi).exp();
                    sum += *x
                }

                let exp = (*mi_1 - mi).exp();
                let exp_mut_di_1 = *di_1 * exp;
                let di = exp_mut_di_1 + sum;

                // 更新 m, l
                *mi_1 = mi;
                *di_1 = di;

                for (j, oi) in oi_.iter_mut().enumerate() {
                    let xv = x
                        .iter()
                        .enumerate()
                        .map(|(i, x)| *x * vj[i * d + j])
                        .sum::<T>();
                    *oi = *oi * exp + xv
                }
            }
        }
        for it in 0..bn {
            let oi = &mut oi_[it * d..][..d];
            let iq = iqb * bn + it;
            let di_1 = &mut di_1_array[it];
            if iq >= n {
                break;
            }
            // 将 oi 写入 o
            oi.iter_mut().for_each(|oi_| *oi_ = *oi_ / *di_1);
            let o = unsafe { from_raw_parts_mut(o.byte_offset(o_strides.offset(head, iq)), d) };
            o.copy_from_slice(oi);
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
            types::F32 => self.test_compute_cpu_::<f32>(reqs),
            types::F64 => self.test_compute_cpu_::<f64>(reqs),
            others => panic!("Unsupported data type {others}"),
        }
    }

    fn test_compute_cpu_<T: Float + FromPrimitive + AddAssign + Sum<T>>(
        &self,
        reqs: &mut [super::test::Attention],
    ) {
        use super::{
            Strides2D,
            test::{Attention, destruct},
        };
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        let &Self {
            h,
            kvh,
            tile_seq,
            tile_ctx,
            ..
        } = self;
        // 生成发送给 kernel 的配置
        let cfg = self.to_kernel_cfg();
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
        // 生成 workspace
        let req_memory = reqs
            .iter()
            .map(|req| {
                let n = req.q.shape()[1];
                let pos = req.pos;
                let s = pos + n;
                // 注意力掩码
                let mask = (0..n * s)
                    .map(|i| i % s <= s - n + i / s)
                    .collect::<Box<_>>();
                mask
            })
            .collect::<Box<_>>();
        // 为每个请求的每个头生成 block
        let reqs = reqs
            .iter()
            .zip(&req_memory)
            .scan(0, |start, (req, mem)| {
                let pages_start = *start as _;
                let n = req.q.shape()[1];
                Some(KernelReq {
                    q: req.q.get().as_ptr().cast(),
                    q_strides: {
                        destruct!([head, seq, _] = req.q.strides());
                        Strides2D { head, seq }
                    },
                    k: req.k.get().as_ptr().cast(),
                    k_strides: {
                        destruct!([head, seq, _] = req.k.strides());
                        Strides2D { head, seq }
                    },
                    v: req.v.get().as_ptr().cast(),
                    v_strides: {
                        destruct!([head, seq, _] = req.v.strides());
                        Strides2D { head, seq }
                    },
                    pages_start,
                    kv_strides: {
                        destruct!([seq, _, head, _] = req.cache.strides());
                        Strides2D { head, seq }
                    },
                    o: req.o.get().as_ptr().cast_mut().cast(),
                    o_strides: {
                        destruct!([head, seq, _] = req.o.strides());
                        Strides2D { head, seq }
                    },
                    mask: mem.as_ptr(),
                    n,
                    s: n + req.pos,
                })
            })
            .collect::<Box<_>>();
        (0..reqs.len())
            .flat_map(|ireq| (0..kvh).map(move |head| [ireq, head]))
            .collect::<Box<_>>()
            .into_par_iter()
            .for_each(|&[ireq, head]| cache_concat_block(cfg, &cache_pages, &reqs, ireq, head));
        (0..reqs.len())
            .flat_map(|ireq| (0..h).map(move |head| [ireq, head]))
            .collect::<Box<_>>()
            .into_par_iter()
            .for_each(|&[ireq, head]| {
                let mut shared = vec![T::zero(); self.shared_elements()];
                flash_attn_block(cfg, &cache_pages, &reqs, ireq, head, tile_seq, &mut shared)
            });
    }
}
