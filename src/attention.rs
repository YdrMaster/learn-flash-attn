use any_tensor::digit_layout::types;
use std::iter::zip;

macro_rules! destruct {
    ($pat:pat = $slice:expr) => {
        let &$pat = &*$slice else {
            panic!("Ndim mismatch ( = {})", $slice.len())
        };
    };
}

pub fn flash_attention(
    q: Tensor<&[u8]>,
    k: Tensor<&[u8]>,
    v: Tensor<&[u8]>,
    mut o: Tensor<&mut [u8]>,
    mut kc: Tensor<&mut [u8]>,
    mut vc: Tensor<&mut [u8]>,
    pos: usize,
    tile_seq: usize, // = tile q  = bn
    tile_ctx: usize, // = tile kv = bs
) {
    // 对齐数据类型
    let dt = distinct(&[q.dt(), k.dt(), v.dt(), o.dt(), kc.dt(), vc.dt()]).unwrap();
    assert_eq!(dt, types::F64);
    // 解构形状维度
    destruct!([h_q, n_q, d_q] = q.shape());
    destruct!([h_o, n_o, d_o] = o.shape());
    destruct!([kvh_k, n_k, d_k] = k.shape());
    destruct!([kvh_v, n_v, d_v] = v.shape());
    destruct!([kvh_kc, buf_k, d_kc] = kc.shape());
    destruct!([kvh_vc, buf_v, d_vc] = vc.shape());
    // 对齐张量形状
    let h = distinct(&[h_q, h_o]).unwrap();
    let kvh = distinct(&[kvh_k, kvh_v, kvh_kc, kvh_vc]).unwrap();
    let n = distinct(&[n_q, n_k, n_v, n_o]).unwrap();
    let s = pos + n;
    let d = distinct(&[d_q, d_k, d_v, d_o, d_kc, d_vc]).unwrap();
    // kv 以页为单位分配
    assert_eq!(s % tile_ctx, 0);
    assert!(buf_k >= s && buf_v >= s);
    // 计算标量参数
    assert_eq!(h % kvh, 0);
    let g = h / kvh;
    let scale = (d as f64).sqrt().recip();
    // 生成 causal mask
    let mask_data = (0..n * s)
        .map(|i| i % s <= s - n + i / s)
        .collect::<Vec<_>>();
    let mask = Tensor::from_dim_slice(types::Bool, [n, s]).map(|_| erase_ty(&mask_data));
    // global
    let mut l = vec![0.; h * s]; // 注意力分母
    let mut m = vec![f64::NEG_INFINITY; h * s]; // 最大值缓存
    // 连接 kv cache
    cache_concat(&k, &v, &mut kc, &mut vc, pos);
    // 计算注意力
    for i in 0..h {
        // local
        let mut qi = vec![0.; tile_seq * d]; // 暂存 q
        let mut kj = vec![0.; tile_ctx * d]; // 暂存 k
        let mut vj = vec![0.; tile_ctx * d]; // 暂存 v
        let mut x = vec![0.; tile_ctx * tile_seq]; // 注意力分数
        // block 之间完全无关，可以以任意方式并行
        FlashAttentionBlock {
            q: &q.as_deref().transform(|l| l.index(0, i)),
            k: &kc.as_deref().transform(|l| l.index(0, i / g)),
            v: &vc.as_deref().transform(|l| l.index(0, i / g)),
            o: &mut o.as_deref_mut().transform(|l| l.index(0, i)),
            mask: &mask.as_deref(),
            l: &mut l[i * s..][..s],
            m: &mut m[i * s..][..s],
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

/// NOTICE GQA 需要发射此 kernel，MHA 可以合并到计算
fn cache_concat(
    k: &Tensor<&[u8]>,
    v: &Tensor<&[u8]>,
    kc: &mut Tensor<&mut [u8]>,
    vc: &mut Tensor<&mut [u8]>,
    pos: usize,
) {
    debug_assert!(matches!(
        distinct(&[k.dt(), v.dt(), kc.dt(), vc.dt()]),
        Some(types::F64)
    ));

    destruct!([kvh_k, n_k, d_k] = k.shape());
    destruct!([kvh_v, n_v, d_v] = v.shape());
    destruct!([kvh_kc, buf_k, d_kc] = kc.shape());
    destruct!([kvh_vc, buf_v, d_vc] = vc.shape());

    let kvh = distinct(&[kvh_k, kvh_v, kvh_kc, kvh_vc]).unwrap();
    let n = distinct(&[n_k, n_v]).unwrap();
    let s = pos + n;
    debug_assert!(distinct(&[d_k, d_v, d_kc, d_vc]).is_some());
    assert!(buf_k >= s && buf_v >= s);

    for i in 0..kvh {
        let k = k.as_deref().transform(|l| l.index(0, i));
        let v = v.as_deref().transform(|l| l.index(0, i));
        let mut kc = kc.as_deref_mut().transform(|l| l.index(0, i));
        let mut vc = vc.as_deref_mut().transform(|l| l.index(0, i));

        for i in 0..n {
            let k = array::<f64>(k.as_deref().transform(|l| l.index(0, i)));
            let v = array::<f64>(v.as_deref().transform(|l| l.index(0, i)));
            let kc = array_mut::<f64>(kc.as_deref_mut().transform(|l| l.index(0, pos + i)));
            let vc = array_mut::<f64>(vc.as_deref_mut().transform(|l| l.index(0, pos + i)));
            kc.copy_from_slice(k);
            vc.copy_from_slice(v);
        }
    }
}

struct FlashAttentionBlock<'a> {
    /// shape = {n, d}
    q: &'a Tensor<&'a [u8]>,
    /// shape = {s, d}
    k: &'a Tensor<&'a [u8]>,
    /// shape = {s, d}
    v: &'a Tensor<&'a [u8]>,
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
            q,
            k,
            v,
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
                    let k = array::<f64>(k.as_deref().transform(|l| l.index(0, ikv)));
                    let v = array::<f64>(v.as_deref().transform(|l| l.index(0, ikv)));
                    ikv += bn;

                    kj[i * d..][..d].copy_from_slice(k);
                    vj[i * d..][..d].copy_from_slice(v);
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
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data)) }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::softmax::test::safe_softmax;

    #[test]
    fn test_flash_attention() {
        const H: usize = 32;
        const KVH: usize = 4;
        const N: usize = 7;
        const S: usize = 4096;
        const P: usize = S - N;
        const D: usize = 2048;

        let q = Tensor::from_dim_slice(types::F64, [H, N, D]);
        let o = Tensor::from_dim_slice(types::F64, [H, N, D]);
        let k = Tensor::from_dim_slice(types::F64, [KVH, N, D]);
        let v = Tensor::from_dim_slice(types::F64, [KVH, N, D]);
        let kc = Tensor::from_dim_slice(types::F64, [KVH, S, D]);
        let vc = Tensor::from_dim_slice(types::F64, [KVH, S, D]);

        let q_data = random_data(H * N * D);
        let k_data = random_data(KVH * N * D);
        let v_data = random_data(KVH * N * D);
        let kc_data = random_data(KVH * S * D);
        let vc_data = random_data(KVH * S * D);

        let q = q.as_ref().map(|_| erase_ty(&q_data));
        let k = k.as_ref().map(|_| erase_ty(&k_data));
        let v = v.as_ref().map(|_| erase_ty(&v_data));
        let kc = kc.as_ref().map(|_| erase_ty(&v_data));
        let vc = vc.as_ref().map(|_| erase_ty(&v_data));

        // 计算标准 attention
        let mut ans = vec![0.0f64; H * N * D];
        let mut kc_ans = kc_data.clone();
        let mut vc_ans = vc_data.clone();
        attention(
            q.as_deref(),
            k.as_deref(),
            v.as_deref(),
            o.as_ref().map(|_| erase_ty_mut(&mut ans)),
            kc.as_ref().map(|_| erase_ty_mut(&mut kc_ans)),
            vc.as_ref().map(|_| erase_ty_mut(&mut vc_ans)),
            P,
        );

        // 计算 flash attention
        let mut res = vec![0.0f64; H * N * D];
        let mut kc_res = kc_data;
        let mut vc_res = vc_data;
        flash_attention(
            q,
            k,
            v,
            o.as_ref().map(|_| erase_ty_mut(&mut res)),
            kc.as_ref().map(|_| erase_ty_mut(&mut kc_res)),
            vc.as_ref().map(|_| erase_ty_mut(&mut vc_res)),
            P,
            4,
            32,
        );

        for (ans, res) in zip(ans, res) {
            let e_abs = (ans - res).abs();
            assert!(
                e_abs < 1e3 * f64::EPSILON,
                "err = {e_abs:.3e} {}x",
                e_abs / f64::EPSILON
            )
        }
    }

    fn random_data(n: usize) -> Vec<f64> {
        (0..n)
            .map(|_| (rand::random::<f64>() - 0.5) * 10.)
            .collect()
    }

    fn erase_ty_mut<T: Copy>(data: &mut [T]) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), size_of_val(data)) }
    }

    /// 多头注意力计算
    pub fn attention(
        q: Tensor<&[u8]>,
        k: Tensor<&[u8]>,
        v: Tensor<&[u8]>,
        mut o: Tensor<&mut [u8]>,
        mut kc: Tensor<&mut [u8]>,
        mut vc: Tensor<&mut [u8]>,
        pos: usize,
    ) {
        // 对齐数据类型
        let dt = distinct(&[q.dt(), k.dt(), v.dt(), o.dt(), kc.dt(), vc.dt()]).unwrap();
        assert_eq!(dt, types::F64);
        // 解构形状维度
        destruct!([h_q, n_q, d_q] = q.shape());
        destruct!([h_o, n_o, d_o] = o.shape());
        destruct!([kvh_k, n_k, d_k] = k.shape());
        destruct!([kvh_v, n_v, d_v] = v.shape());
        destruct!([kvh_kc, buf_k, d_kc] = kc.shape());
        destruct!([kvh_vc, buf_v, d_vc] = vc.shape());
        // 对齐张量形状
        let h = distinct(&[h_q, h_o]).unwrap();
        let kvh = distinct(&[kvh_k, kvh_v, kvh_kc, kvh_vc]).unwrap();
        let n = distinct(&[n_q, n_k, n_v, n_o]).unwrap();
        let s = pos + n;
        let d = distinct(&[d_q, d_k, d_v, d_o, d_kc, d_vc]).unwrap();
        assert!(buf_k >= s && buf_v >= s);
        // 计算标量参数
        assert_eq!(h % kvh, 0);
        let g = h / kvh;
        let scale = (d as f64).sqrt().recip();
        // 连接 kv cache
        cache_concat(&k, &v, &mut kc, &mut vc, pos);
        // 计算注意力
        let mut score = vec![0.; s];
        for i in 0..h {
            let q = q.as_deref().transform(|l| l.index(0, i));
            let k = kc.as_deref().transform(|l| l.index(0, i / g));
            let v = vc.as_deref().transform(|l| l.index(0, i / g));
            let mut o = o.as_deref_mut().transform(|l| l.index(0, i));

            for i in 0..n {
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
