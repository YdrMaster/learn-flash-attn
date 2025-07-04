use super::{Attention, Tensor, array, destruct, distinct, erase_ty};
use crate::attention::FlashAttnCfg;
use any_tensor::digit_layout::types;
use std::{iter::zip, rc::Rc};

impl FlashAttnCfg {
    pub fn compute_cpu(&self, reqs: &mut [Attention]) {
        let &Self {
            h,
            kvh,
            d,
            tile_seq,
            tile_ctx,
        } = self;

        let mut blocks = Vec::new();

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
            distinct(&[h, h_q, h_o]).unwrap();
            distinct(&[kvh, kvh_k, kvh_v, kvh_c]).unwrap();
            distinct(&[d, d_q, d_k, d_v, d_o, d_c]).unwrap();
            let n = distinct(&[n_q, n_k, n_v, n_o]).unwrap();
            let s = *pos + n;
            assert!(buf >= s);
            // 计算标量参数
            assert_eq!(h % kvh, 0);
            let g = h / kvh;
            let scale = (d as f64).sqrt().recip();
            // kv 以页为单位分配
            destruct!([sbuf, skv, sh, sd] = cache.strides());
            assert_eq!(sd, dt.nbytes() as isize);
            assert_eq!(s % tile_ctx, 0);
            let pages = (0..s / tile_ctx)
                .map(|i| {
                    let t = cache.as_ref().transform(|l| l.index(0, i * tile_ctx));
                    unsafe { t.get().as_ptr().byte_offset(t.offset()).cast_mut().cast() }
                })
                .collect::<Box<_>>();
            let pages = CachePages {
                pages,
                strides: [sbuf, skv, sh],
                bs: tile_ctx,
                dh: d,
            };
            // 生成 causal mask
            let mask_data = (0..n * s)
                .map(|i| i % s <= s - n + i / s)
                .collect::<Rc<_>>();
            let mask = Tensor::from_dim_slice(types::Bool, [n, s]).map(|_| erase_ty(&mask_data));
            // 连接 kv cache
            pages.concat(&k, &v, *pos);
            let pages = Rc::new(pages);
            // 计算注意力
            for head in 0..h {
                // global
                let q = q.as_deref().transform(|l| l.index(0, head));
                let mut o = o.as_deref_mut().transform(|l| l.index(0, head));
                let l = vec![0.; s]; // 注意力分母
                let m = vec![f64::NEG_INFINITY; s]; // 最大值缓存
                // local
                let qi = vec![0.; tile_seq * d]; // 暂存 q
                let kj = vec![0.; tile_ctx * d]; // 暂存 k
                let vj = vec![0.; tile_ctx * d]; // 暂存 v
                let x = vec![0.; tile_ctx * tile_seq]; // 注意力分数
                // block 之间完全无关，可以以任意方式并行
                let blk = FlashAttentionBlock {
                    head: head / g,
                    pages: pages.clone(),

                    q: Matrix::<*const f64>::from_tensor(&q),
                    o: Matrix::<*mut f64>::from_tensor(&mut o),
                    mask: Matrix::<*const bool>::from_tensor(&mask),
                    l,
                    m,
                    qi,
                    kj,
                    vj,
                    x,
                    n,
                    s,
                    d,
                    bn: tile_seq,
                    bs: tile_ctx,
                    scale,
                };
                blocks.push((mask_data.clone(), blk))
            }
        }

        blocks.reverse(); // 验证执行顺序是无关的
        for block in blocks {
            block.1.launch()
        }
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
}

struct FlashAttentionBlock {
    head: usize,
    pages: Rc<CachePages>,

    q: Matrix<*const f64>,
    /// shape = {n, d}
    o: Matrix<*mut f64>,
    /// shape = {n, s}
    mask: Matrix<*const bool>,
    /// shape = {s}
    l: Vec<f64>,
    /// shape = {s}
    m: Vec<f64>,
    /// shape = {bn, d}
    qi: Vec<f64>,
    /// shape = {bs, d}
    kj: Vec<f64>,
    /// shape = {bs, d}
    vj: Vec<f64>,
    /// shape = {bn, bs}
    x: Vec<f64>,

    n: usize,
    s: usize,
    d: usize,
    bn: usize,
    bs: usize,
    scale: f64,
}

struct Matrix<T> {
    ptr: T,
    ld: isize,
}

impl<T> Matrix<*const T> {
    fn from_tensor(t: &Tensor<&[u8]>) -> Self {
        destruct!([m, n] = t.strides());
        assert_eq!(n, t.dt().nbytes() as isize);
        Self {
            ptr: unsafe { t.get().as_ptr().byte_offset(t.offset()).cast() },
            ld: m / n,
        }
    }

    fn row(&self, i: usize, d: usize) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.offset(i as isize * self.ld), d) }
    }
}

impl<T> Matrix<*mut T> {
    fn from_tensor(t: &mut Tensor<&mut [u8]>) -> Self {
        destruct!([m, n] = t.strides());
        assert_eq!(n, t.dt().nbytes() as isize);
        Self {
            ptr: unsafe { t.get_mut().as_mut_ptr().byte_offset(t.offset()).cast() },
            ld: m / n,
        }
    }

    fn row(&self, i: usize, d: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.offset(i as isize * self.ld), d) }
    }
}

impl FlashAttentionBlock {
    fn launch(self) {
        let Self {
            head,
            pages,
            q,
            o,
            mask,
            mut l,
            mut m,
            mut qi,
            mut kj,
            mut vj,
            mut x,
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

                    kj[i * d..][..d].copy_from_slice(&k);
                    vj[i * d..][..d].copy_from_slice(&v);
                    i += bn;
                }
            }
            // thread
            for it in 0..bn {
                let qi = &mut qi[it * d..][..d];
                let x = &mut x[it * bs..][..bs];

                for iqb in 0..n.div_ceil(bn) {
                    // 每个线程计算 q 的一行
                    let iq = iqb * bn + it;
                    if iq >= n {
                        break;
                    }
                    // locate data
                    let q = q.row(iq, d);
                    let o = o.row(iq, d);
                    let mask = mask.row(iq, s);
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
