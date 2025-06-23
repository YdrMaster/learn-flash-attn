use any_tensor::digit_layout::types;
use cuda::{Device, Ptx, memcpy_d2h};
use std::{
    ffi::{c_int, c_uint},
    iter::zip,
    ops::Range,
};

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
                    for (x, i) in zip(&mut x, tiled.clone()) {
                        let k = array::<f64>(k.as_deref().transform(|l| l.index(0, i)));
                        *x = zip(q, k).map(|(q, k)| q * k).sum::<f64>() * scale
                    }

                    let mi = x.iter().copied().fold(mi_1, f64::max);
                    for x in &mut *x {
                        *x = (*x - mi).exp()
                    }

                    let mut exp = di_1 * (mi_1 - mi).exp();
                    let di = exp + x.iter().sum::<f64>();

                    mi_1 = mi;
                    di_1 = di;

                    exp /= di;
                    x.iter_mut().for_each(|x| *x /= di);

                    let mut o_local = vec![0.; o.len()];
                    for (x, i) in zip(x, tiled) {
                        let v = array::<f64>(v.as_deref().transform(|l| l.index(0, i)));
                        zip(v, &mut o_local).for_each(|(v, o)| *o += x * v)
                    }

                    for (o, o_) in zip(&mut *o, o_local) {
                        *o = *o * exp + o_
                    }
                }
            }
        }
    }
}

pub fn cuda(
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

    assert_eq!(n, s);
    assert_eq!(h, kvh);
    // let g = h / kvh;
    let scale = (d as f64).sqrt().recip();

    const TEXT: &str = r#"
#include <math_constants.h>

extern "C" __global__ void forward_kernel(
const double* Q, const double* K, const double* V,
const int N, const int d,
const int Tc, const int Tr, const int Bc, const int Br,
const double softmax_scale,
double* l, double *m, double* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ double sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    double* Qi = sram;
    double* Kj = &sram[tile_size];
    double* Vj = &sram[tile_size * 2];
    double* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            double row_m_prev = m[lm_offset + (Br * i) + tx];
            double row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            double row_m = -CUDART_INF_F;
            for (int y = 0; y < Bc; y++) {
                double sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            double row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            double row_m_new = max(row_m_prev, row_m);
            double row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                double pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}"#;

    assert!(cuda::init().is_ok());
    let device = Device::new(0);
    let (ptx, log) = Ptx::compile(TEXT, device.compute_capability());
    let Ok(ptx) = ptx else { panic!("{log}") };

    device.context().apply(|ctx| {
        let module = ctx.load(&ptx);
        let kernel = module.get_kernel(c"forward_kernel");

        let q = q.map(|s| ctx.from_host(s));
        let k = k.map(|s| ctx.from_host(s));
        let v = v.map(|s| ctx.from_host(s));
        let o_ = o.as_deref().map(|s| ctx.malloc::<u8>(s.len()));
        let l = any_tensor::Tensor::new(types::F64, [h, s]).map(|len| ctx.malloc::<u8>(len));
        let m = any_tensor::Tensor::new(types::F64, [h, s]).map(|len| ctx.malloc::<u8>(len));

        let params = cuda::params![
            q.get().as_ptr(),
            k.get().as_ptr(),
            v.get().as_ptr(),
            n as c_int,
            d as c_int,
            n.div_ceil(tile_ctx) as c_int,
            n.div_ceil(tile_seq) as c_int,
            tile_ctx as c_int,
            tile_seq as c_int,
            scale,
            l.get().as_ptr(),
            m.get().as_ptr(),
            o_.get().as_ptr()
        ];

        ctx.stream()
            .launch(
                &kernel,
                (
                    h as c_uint,
                    tile_ctx as c_uint,
                    (3 * tile_ctx * d + tile_ctx * tile_seq) * size_of::<f64>(),
                ),
                &*params.to_ptrs(),
            )
            .synchronize();

        memcpy_d2h(o.get_mut(), o_.get());
    });
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
        const KVH: usize = 32;
        const N: usize = 256;
        const S: usize = 256;
        const D: usize = 32;

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
        flash_attention(
            q.as_deref(),
            k.as_deref(),
            v.as_deref(),
            o.as_ref().map(|_| erase_ty_mut(&mut res)),
            32,
            32,
        );

        let mut res_ = vec![0.0f64; H * N * D];
        cuda(q, k, v, o.as_ref().map(|_| erase_ty_mut(&mut res_)), 32, 32);

        for (i, (ans, res)) in zip(ans, res_).enumerate() {
            let e_abs = (ans - res).abs();
            if e_abs > 1e-3 {
                println!("i = {i} {ans:.3e} != {res:.3e} err = {e_abs}")
            }
            // assert!(
            //     e_abs < 1e3 * f64::EPSILON,
            //     "err = {e_abs:.3e} {}x",
            //     e_abs / f64::EPSILON
            // )
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
