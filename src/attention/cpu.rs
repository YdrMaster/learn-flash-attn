use super::{Attention, destruct};
use crate::attention::{
    FlashAttnCfg, cache_concat,
    kernel::{KVPage, KernelCfg, KernelReq, Strides2D},
};
use rand::seq::SliceRandom;
use std::{
    iter::zip,
    ptr::copy_nonoverlapping,
    slice::{from_raw_parts, from_raw_parts_mut},
};

impl FlashAttnCfg {
    pub fn compute_cpu(&self, reqs: &mut [Attention]) {
        // 连接 kv cache
        for req in &mut *reqs {
            cache_concat(&req.k, &req.v, &mut req.cache, req.pos)
        }
        let &Self {
            h,
            tile_seq,
            tile_ctx,
            ..
        } = self;
        // 生成发送给 kernel 的配置
        let cfg = self.to_kernel_cfg();
        // 生成所有页指针
        let cache_pages = reqs
            .iter()
            .flat_map(|req| {
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
                        k: unsafe { base.byte_offset(k).cast() },
                        v: unsafe { base.byte_offset(v).cast() },
                    }
                })
            })
            .collect::<Vec<_>>();
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
                // 注意力分母
                let l = vec![0.; h * s];
                // 最大值缓存
                let m = vec![f64::NEG_INFINITY; h * s];
                (mask, l, m)
            })
            .collect::<Vec<_>>();
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
                    mask: mem.0.as_ptr(),
                    l: mem.1.as_ptr().cast_mut(),
                    m: mem.2.as_ptr().cast_mut(),
                    n,
                    s: n + req.pos,
                })
            })
            .collect::<Vec<_>>();
        // 生成线程块序号
        let mut indices = (0..reqs.len())
            .flat_map(|ireq| (0..h).map(move |head| [ireq, head]))
            .collect::<Vec<_>>();
        // 以任意顺序执行
        indices.shuffle(&mut rand::rng());
        for [ireq, head] in indices {
            let mut shared = vec![0.; self.shared_elements()];
            launch_block(cfg, &cache_pages, &reqs, ireq, head, tile_seq, &mut shared)
        }
    }
}

fn launch_block(
    cfg: KernelCfg,
    cache_pages: &[KVPage],
    reqs: &[KernelReq],
    ireq: usize,
    head: usize,
    bn: usize,
    shared: &mut [f64],
) {
    let KernelCfg { g, d, bs, scale } = cfg;
    let &KernelReq {
        q,
        q_strides,
        pages_start,
        kv_strides,
        o,
        o_strides,
        mask,
        l,
        m,
        n,
        s,
    } = &reqs[ireq];
    let pages = &cache_pages[pages_start..][..s.div_ceil(bs)];
    // 划分 shared memory
    let (qi, shared) = shared.split_at_mut(bn * d);
    let (kj, shared) = shared.split_at_mut(bs * d);
    let (vj, x) = shared.split_at_mut(bs * d);
    assert_eq!(x.len(), bn * bs);
    // 定位每个线程块的 m 和 l，长度 s
    let m = unsafe { from_raw_parts_mut(m.add(head * s), s) };
    let l = unsafe { from_raw_parts_mut(l.add(head * s), s) };
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
            let qi = &mut qi[it * d..][..d];
            let x = &mut x[it * bs..][..bs];
            // seq 方向迭代
            for iqb in 0..n.div_ceil(bn) {
                let iq = iqb * bn + it;
                if iq >= n {
                    break;
                }
                // locate data
                let q = unsafe { from_raw_parts(q.byte_offset(q_strides.offset(head, iq)), d) };
                let o = unsafe { from_raw_parts_mut(o.byte_offset(o_strides.offset(head, iq)), d) };
                // load data
                qi.copy_from_slice(q);
                let mask = unsafe { from_raw_parts(mask.add(iq * s + ikvb * bs), bs) };

                let mi_1 = m[iq];
                let di_1 = l[iq];

                // score = q @ k^T / √d
                let mut mi = mi_1;
                for (i, (mask, x)) in zip(mask, &mut *x).enumerate() {
                    if !*mask {
                        *x = f64::NEG_INFINITY
                    } else {
                        let qk = zip(&*qi, &kj[i * d..][..d])
                            .map(|(q, k)| q * k)
                            .sum::<f64>()
                            * scale as f64;
                        *x = qk;
                        if qk > mi {
                            mi = qk
                        }
                    }
                }

                let mut sum = 0.;
                for x in &mut *x {
                    *x = (*x - mi).exp();
                    sum += *x
                }

                let exp = di_1 * (mi_1 - mi).exp();
                let di = exp + sum;
                // 更新 m, l
                m[iq] = mi;
                l[iq] = di;

                for (j, o) in o.iter_mut().enumerate() {
                    let xv = x
                        .iter()
                        .enumerate()
                        .map(|(i, x)| *x * vj[i * d + j])
                        .sum::<f64>();
                    *o = (*o * exp + xv) / di
                }
            }
        }
    }
}
