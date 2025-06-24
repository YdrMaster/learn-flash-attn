use any_tensor::digit_layout::types;
use std::{iter::zip, ops::Range};

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
    tile_seq: usize,
    tile_ctx: usize,
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

    // 处理每个头
    for i in 0..h {
        let q = q.as_deref().transform(|layout| layout.index(0, i));
        let k = k.as_deref().transform(|layout| layout.index(0, i / g));
        let v = v.as_deref().transform(|layout| layout.index(0, i / g));
        let mut o = o.as_deref_mut().transform(|layout| layout.index(0, i));

        // 处理每个 token
        for i in 0..n.div_ceil(tile_seq) {
            for i in tile(i, n, tile_seq) {
                let q = array::<f64>(q.as_deref().transform(|l| l.index(0, i)));
                let o = array_mut::<f64>(o.as_deref_mut().transform(|l| l.index(0, i)));
                o.fill(0.);

                let len = s - n + i + 1;
                let mut mi_1 = f64::NEG_INFINITY;
                let mut di_1 = 0.;

                for i in 0..len.div_ceil(tile_ctx) {
                    let tiled = tile(i, len, tile_ctx);
                    let mut x = vec![0.; tiled.len()];

                    // score = q @ k^T / √d
                    let mut mi = mi_1;
                    for (x, i) in zip(&mut x, tiled.clone()) {
                        let k = array::<f64>(k.as_deref().transform(|l| l.index(0, i)));
                        *x = zip(q, k).map(|(q, k)| q * k).sum::<f64>() * scale;
                        mi = f64::max(mi, *x)
                    }

                    let mut sum = 0.;
                    for x in &mut *x {
                        *x = (*x - mi).exp();
                        sum += *x
                    }

                    let mut exp = di_1 * (mi_1 - mi).exp();
                    let di = exp + sum;

                    mi_1 = mi;
                    di_1 = di;

                    exp /= di;
                    x.iter_mut().for_each(|x| *x /= di);

                    let mut xv = vec![0.; o.len()];
                    for (i, x) in zip(tiled, x) {
                        let v = array::<f64>(v.as_deref().transform(|l| l.index(0, i)));
                        zip(&mut xv, v).for_each(|(xv, v)| *xv += x * v)
                    }
                    for (o, xv) in zip(&mut *o, xv) {
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

/// 产生分块的序号范围
fn tile(i: usize, total: usize, tile: usize) -> Range<usize> {
    i * tile..((i + 1) * tile).min(total)
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
        const D: usize = 256;

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
                score[len..].fill(0.);
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
