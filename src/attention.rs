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
    tile_seq: usize, // = tile q  = Br
    tile_ctx: usize, // = tile kv = Bc
) {
    assert_eq!(q.dt(), types::F64);
    assert_eq!(k.dt(), types::F64);
    assert_eq!(v.dt(), types::F64);
    assert_eq!(o.dt(), types::F64);

    destruct!([h_q, n_q, d_q] = q.shape());
    destruct!([h_o, n_o, d_o] = o.shape());
    destruct!([kvh_k, seq_k, d_k] = k.shape());
    destruct!([kvh_v, seq_v, d_v] = v.shape());

    let h = distinct(&[h_q, h_o]).unwrap();
    let kvh = distinct(&[kvh_k, kvh_v]).unwrap();
    let n = distinct(&[n_q, n_o]).unwrap();
    let s = distinct(&[seq_k, seq_v]).unwrap();
    let d = distinct(&[d_q, d_k, d_v, d_o]).unwrap();

    assert_eq!(h % kvh, 0);
    let g = h / kvh;
    let scale = (d as f64).sqrt().recip();

    assert_eq!(tile_ctx, tile_seq);
    assert_eq!(s % tile_ctx, 0);

    // global
    let mut l = vec![0.; h * s]; // 注意力分母
    let mut m = vec![f64::NEG_INFINITY; h * s]; // 最大值缓存

    // causal mask
    let mask = Tensor::from_dim_slice(types::Bool, &[n, s]).map(|len| {
        (0..len)
            .map(|i| if i % s <= s - n + i / s { 1u8 } else { 0u8 })
            .collect::<Vec<_>>()
    });

    for i in 0..h {
        // local
        let mut qi = vec![0.; tile_ctx * d]; // 暂存 q
        let mut kj = vec![0.; tile_ctx * d]; // 暂存 k
        let mut vj = vec![0.; tile_ctx * d]; // 暂存 v
        let mut x = vec![0.; tile_ctx * tile_seq]; // 注意力分数
        // block 之间完全无关，可以以任意方式并行
        FlashAttentionBlock {
            q: &q.as_deref().transform(|layout| layout.index(0, i)),
            k: &k.as_deref().transform(|layout| layout.index(0, i / g)),
            v: &v.as_deref().transform(|layout| layout.index(0, i / g)),
            o: &mut o.as_deref_mut().transform(|layout| layout.index(0, i)),
            mask: &mask.as_deref(),
            l: &mut l[i * s..][..s],
            m: &mut m[i * s..][..s],
            qi: &mut qi,
            kj: &mut kj,
            vj: &mut vj,
            x: &mut x,
            n,
            d,
            b: tile_ctx,
            tr: n.div_ceil(tile_seq),
            tc: s.div_ceil(tile_ctx),
            scale,
        }
        .launch()
    }
}

struct FlashAttentionBlock<'a> {
    q: &'a Tensor<&'a [u8]>,
    k: &'a Tensor<&'a [u8]>,
    v: &'a Tensor<&'a [u8]>,
    o: &'a mut Tensor<&'a mut [u8]>,
    mask: &'a Tensor<&'a [u8]>,
    l: &'a mut [f64],
    m: &'a mut [f64],
    qi: &'a mut [f64],
    kj: &'a mut [f64],
    vj: &'a mut [f64],
    x: &'a mut [f64],

    n: usize,
    d: usize,
    b: usize,
    tr: usize,
    tc: usize,
    scale: f64,
}

impl FlashAttentionBlock<'_> {
    fn launch(self) {
        let Self {
            q,    // [n, d]
            k,    // [s, d]
            v,    // [s, d]
            o,    // [n, d]
            mask, // [n, s]
            l,    // [s]
            m,    // [s]
            qi,   // [b, d]
            kj,   // [b, d]
            vj,   // [b, d]
            x,    // [b, b]
            n,
            d,
            b,
            tr,
            tc,
            scale,
        } = self;

        for j in 0..tc {
            // thread
            for tx in 0..b {
                // 每个线程拷贝 k/v 的一行，拷贝整个 kv block 到 local memory
                let kj = &mut kj[tx * d..][..d];
                let vj = &mut vj[tx * d..][..d];

                let k = array::<f64>(k.as_deref().transform(|l| l.index(0, j * b + tx)));
                let v = array::<f64>(v.as_deref().transform(|l| l.index(0, j * b + tx)));

                kj.copy_from_slice(k);
                vj.copy_from_slice(v);
            }
            // thread
            for tx in 0..b {
                let qi = &mut qi[tx * d..][..d];
                let x = &mut x[tx * b..][..b];

                for i in 0..tr {
                    if i * b + tx >= n {
                        break;
                    }
                    // load q
                    let q = array::<f64>(q.as_deref().transform(|l| l.index(0, i * b + tx)));
                    qi.copy_from_slice(q);
                    // locate o
                    let o =
                        array_mut::<f64>(o.as_deref_mut().transform(|l| l.index(0, i * b + tx)));
                    // locate mask
                    let mask = array::<u8>(mask.as_deref().transform(|l| l.index(0, i * b + tx)));
                    let mask = &mask[j * b..][..b];

                    let mi_1 = m[i * b + tx];
                    let di_1 = l[i * b + tx];

                    // score = q @ k^T / √d
                    let mut mi = mi_1;
                    for (y, (mask, x)) in zip(mask, &mut *x).enumerate() {
                        if *mask == 0 {
                            *x = f64::NEG_INFINITY
                        } else {
                            let kj = &kj[y * d..][..d];
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

                    m[i * b + tx] = mi;
                    l[i * b + tx] = di;

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
    let [ans, tail @ ..] = &*val else {
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::softmax::test::safe_softmax;

    #[test]
    fn test_flash_attention() {
        const H: usize = 32;
        const KVH: usize = 4;
        const N: usize = 7;
        const S: usize = 2048;
        const D: usize = 2048;

        let q = Tensor::from_dim_slice(types::F64, &[H, N, D]);
        let k = Tensor::from_dim_slice(types::F64, &[KVH, S, D]);
        let v = Tensor::from_dim_slice(types::F64, &[KVH, S, D]);
        let o = Tensor::from_dim_slice(types::F64, &[H, N, D]);

        let q_data = random_data(H * N * D);
        let k_data = random_data(KVH * S * D);
        let v_data = random_data(KVH * S * D);

        let q = q.as_ref().map(|_| erase_ty(&q_data));
        let k = k.as_ref().map(|_| erase_ty(&k_data));
        let v = v.as_ref().map(|_| erase_ty(&v_data));

        // 计算标准 attention
        let mut ans = vec![0.0f64; H * N * D];
        attention(
            q.as_deref(),
            k.as_deref(),
            v.as_deref(),
            o.as_ref().as_ref().map(|_| erase_ty_mut(&mut ans)),
        );

        // 计算 flash attention
        let mut res = vec![0.0f64; H * N * D];
        flash_attention(q, k, v, o.as_ref().map(|_| erase_ty_mut(&mut res)), 32, 32);

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

    fn erase_ty<T: Copy>(data: &[T]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data)) }
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
    ) {
        assert_eq!(q.dt(), types::F64);
        assert_eq!(k.dt(), types::F64);
        assert_eq!(v.dt(), types::F64);
        assert_eq!(o.dt(), types::F64);

        destruct!([h_q, n_q, d_q] = q.shape());
        destruct!([h_o, n_o, d_o] = o.shape());
        destruct!([kvh_k, seq_k, d_k] = k.shape());
        destruct!([kvh_v, seq_v, d_v] = v.shape());

        let h = distinct(&[h_q, h_o]).unwrap();
        let kvh = distinct(&[kvh_k, kvh_v]).unwrap();
        assert_eq!(h % kvh, 0);
        let n = distinct(&[n_q, n_o]).unwrap();
        let s = distinct(&[seq_k, seq_v]).unwrap();
        let d = distinct(&[d_q, d_k, d_v, d_o]).unwrap();

        let g = h / kvh;
        let scale = (d as f64).sqrt().recip();
        let mut score = vec![0.; s];
        // 处理每个头
        for i in 0..h {
            let q = q.as_deref().transform(|layout| layout.index(0, i));
            let k = k.as_deref().transform(|layout| layout.index(0, i / g));
            let v = v.as_deref().transform(|layout| layout.index(0, i / g));
            let mut o = o.as_deref_mut().transform(|layout| layout.index(0, i));

            // 处理每个 token
            for i in 0..n {
                let q = array::<f64>(q.as_deref().transform(|layout| layout.index(0, i)));
                let o = array_mut::<f64>(o.as_deref_mut().transform(|layout| layout.index(0, i)));
                // score = q @ k^T / √d
                let len = s - n + i + 1;
                for i in 0..len {
                    let k = array::<f64>(k.as_deref().transform(|layout| layout.index(0, i)));
                    score[i] = zip(q, k).map(|(q, k)| q * k).sum::<f64>() * scale
                }
                // causal softmax
                safe_softmax(&mut score[..len]);
                // o = a @ v // 乘法不连续
                o.fill(0.);
                for i in 0..len {
                    let v = array::<f64>(v.as_deref().transform(|layout| layout.index(0, i)));
                    for (v, o) in zip(v, &mut *o) {
                        *o += score[i] * v
                    }
                }
            }
        }
    }
}
