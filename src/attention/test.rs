use super::FlashAttnCfg;
use crate::softmax::test::safe_softmax;
use any_tensor::digit_layout::{DigitLayout, types};
use num_traits::{Float, FromPrimitive};
use std::{
    iter::{Sum, zip},
    ops::{AddAssign, DivAssign},
};

pub(super) use macros::destruct;

mod macros {
    macro_rules! destruct {
        ($pat:pat = $slice:expr) => {
            let &$pat = &*$slice else {
                panic!("Ndim mismatch ( = {})", $slice.len())
            };
        };
    }
    pub(crate) use destruct;
}

pub(super) struct Attention<'a> {
    pub q: Tensor<&'a [u8]>,
    pub k: Tensor<&'a [u8]>,
    pub v: Tensor<&'a [u8]>,
    pub o: Tensor<&'a mut [u8]>,
    pub cache: Tensor<&'a mut [u8]>,
    pub pos: usize,
}

impl Attention<'_> {
    pub fn dt(&self) -> DigitLayout {
        let Self {
            q, k, v, o, cache, ..
        } = self;
        distinct(&[q.dt(), k.dt(), v.dt(), o.dt(), cache.dt()]).unwrap()
    }
}

pub(super) type Tensor<T> = any_tensor::Tensor<T, 3>;

#[test]
fn test_flash_attention() {
    type Tdata = f64;
    const DT: DigitLayout = types::F64;

    const H: usize = 32;
    const KVH: usize = 8;
    const N: usize = 7;
    const S: usize = 1000;
    const P: usize = S - N;
    const D: usize = 64;

    const FLASH_ATTN: FlashAttnCfg = FlashAttnCfg {
        h: H,
        kvh: KVH,
        d: D,
        tile_seq: 4,
        tile_ctx: 32,
    };

    let q = Tensor::from_dim_slice(DT, [H, N, D]);
    let o = Tensor::from_dim_slice(DT, [H, N, D]);
    let k = Tensor::from_dim_slice(DT, [KVH, N, D]);
    let v = Tensor::from_dim_slice(DT, [KVH, N, D]);
    let cache = Tensor::from_dim_slice(DT, [S, 2, KVH, D]);

    let q_data = random_data::<Tdata>(H * N * D);
    let k_data = random_data::<Tdata>(KVH * N * D);
    let v_data = random_data::<Tdata>(KVH * N * D);
    let cache_data = random_data::<Tdata>(S * 2 * KVH * D);

    let q = q.as_ref().map(|_| erase_ty(&q_data));
    let k = k.as_ref().map(|_| erase_ty(&k_data));
    let v = v.as_ref().map(|_| erase_ty(&v_data));
    let cache = cache.as_ref().map(|_| erase_ty(&v_data));

    // 计算标准 attention
    let mut ans = vec![0.; H * N * D];
    let mut cache_ans = cache_data.clone();
    attention::<Tdata>(
        q.as_deref(),
        k.as_deref(),
        v.as_deref(),
        o.as_ref().map(|_| erase_ty_mut(&mut ans)),
        cache.as_ref().map(|_| erase_ty_mut(&mut cache_ans)),
        P,
    );

    let max_error = 10. * Tdata::EPSILON.sqrt();
    println!("max error: {max_error:e}");

    // 计算 flash attention
    {
        let mut res = vec![0.; H * N * D];
        let mut cache_res = cache_data.clone();
        let mut reqs = [Attention {
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            o: o.as_ref().map(|_| erase_ty_mut(&mut res)),
            cache: cache.as_ref().map(|_| erase_ty_mut(&mut cache_res)),
            pos: P,
        }];
        FLASH_ATTN.test_compute_cpu(&mut reqs);

        let max = zip(ans.clone(), res)
            .chain(zip(cache_ans.clone(), cache_res))
            .map(|(ans, res)| (ans - res).abs())
            .fold(0., Tdata::max);
        println!("CPU: max error = {max:e}");
        assert!(max < max_error, "CPU error mismatch")
    }
    #[cfg(cuda)]
    {
        let mut res = vec![0.; H * N * D];
        let mut cache_res = cache_data.clone();
        let mut reqs = [Attention {
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            o: o.as_ref().map(|_| erase_ty_mut(&mut res)),
            cache: cache.as_ref().map(|_| erase_ty_mut(&mut cache_res)),
            pos: P,
        }];

        cuda::init().unwrap();
        cuda::Device::new(0)
            .context()
            .apply(move |ctx| FLASH_ATTN.test_compute_cuda(&mut reqs, &ctx.stream()));

        let max = zip(ans.clone(), res)
            .chain(zip(cache_ans.clone(), cache_res))
            .map(|(ans, res)| (ans - res).abs())
            .fold(0., Tdata::max);
        println!("GPU: max error = {max:e}");
        assert!(max < max_error, "GPU error mismatch")
    }
}

fn random_data<T: Float + FromPrimitive>(n: usize) -> Vec<T> {
    (0..n)
        .map(|_| T::from_f32((rand::random::<f32>() - 0.5) * 10.).unwrap())
        .collect()
}

/// 连接 cache
fn cache_concat<T: Copy>(
    k: &Tensor<&[u8]>,
    v: &Tensor<&[u8]>,
    cache: &mut Tensor<&mut [u8]>,
    pos: usize,
) {
    distinct(&[k.dt(), v.dt(), cache.dt()]).unwrap();

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
            let k = array::<T>(k.as_deref().transform(|l| l.index(0, i)));
            let v = array::<T>(v.as_deref().transform(|l| l.index(0, i)));
            array_mut::<T>(cache.as_deref_mut().transform(|l| l.index(0, 0))).copy_from_slice(k);
            array_mut::<T>(cache.as_deref_mut().transform(|l| l.index(0, 1))).copy_from_slice(v);
        }
    }
}

/// 多头注意力计算
pub fn attention<T: Float + FromPrimitive + AddAssign + DivAssign + Sum>(
    q: Tensor<&[u8]>,
    k: Tensor<&[u8]>,
    v: Tensor<&[u8]>,
    mut o: Tensor<&mut [u8]>,
    mut cache: Tensor<&mut [u8]>,
    pos: usize,
) {
    // 对齐数据类型
    let _ = distinct(&[q.dt(), k.dt(), v.dt(), o.dt(), cache.dt()]).unwrap();
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
    let scale = T::from_usize(d).unwrap().sqrt().recip();
    // 连接 kv cache
    cache_concat::<T>(&k, &v, &mut cache, pos);
    // 计算注意力
    let k = cache.as_deref().transform(|l| l.index(1, 0));
    let v = cache.as_deref().transform(|l| l.index(1, 1));
    for i in 0..h {
        let q = q.as_deref().transform(|l| l.index(0, i));
        let k = k.as_deref().transform(|l| l.index(1, i / g));
        let v = v.as_deref().transform(|l| l.index(1, i / g));
        let mut o = o.as_deref_mut().transform(|l| l.index(0, i));

        for i in 0..n {
            let mut score = vec![T::zero(); s];
            let q = array::<T>(q.as_deref().transform(|l| l.index(0, i)));
            let o = array_mut::<T>(o.as_deref_mut().transform(|l| l.index(0, i)));
            // score = q @ k^T / √d
            let len = s - n + i + 1;
            for (i, score) in score.iter_mut().enumerate().take(len) {
                let k = array::<T>(k.as_deref().transform(|l| l.index(0, i)));
                *score = zip(q, k).map(|(&q, &k)| q * k).sum::<T>() * scale
            }
            // causal softmax
            safe_softmax(&mut score[..len]);
            // o = a @ v // 乘法不连续
            o.fill(T::zero());
            for (i, &score) in score.iter().enumerate().take(len) {
                let v = array::<T>(v.as_deref().transform(|l| l.index(0, i)));
                for (&v, o) in zip(v, &mut *o) {
                    *o += score * v
                }
            }
        }
    }
}

fn erase_ty<T: Copy>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data)) }
}

fn erase_ty_mut<T: Copy>(data: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), size_of_val(data)) }
}

pub(super) fn distinct<T: Eq + Copy>(val: &[T]) -> Option<T> {
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
