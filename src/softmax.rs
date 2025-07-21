//! READ: <https://zhuanlan.zhihu.com/p/11656282335>

/// 增量 softmax 实现
pub fn online_softmax(data: &mut [f64], block_size: usize) {
    // 一次遍历，同时确定最大值和指数和
    let mut s = S::new(&[]);

    let mut slice = &data[..];
    while slice.len() > block_size {
        let (block, slice_) = slice.split_at(block_size);
        s = S::reduce(s, S::new(block));
        slice = slice_
    }

    let S { max, sum_exp: sum } = S::reduce(s, S::new(slice));

    // 归一化
    for x in data {
        *x = (*x - max).exp() / sum
    }
}

#[derive(Clone, Copy)]
pub(crate) struct S {
    pub max: f64,
    pub sum_exp: f64,
}

impl S {
    pub fn new(data: &[f64]) -> Self {
        let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp = data.iter().map(|x| (x - max).exp()).sum();
        Self { max, sum_exp }
    }

    pub fn reduce(a: Self, b: Self) -> Self {
        use std::cmp::Ordering::{Equal, Greater, Less};
        let [l, g] = match a.max.total_cmp(&b.max) {
            Equal => {
                return Self {
                    max: a.max,
                    sum_exp: a.sum_exp + b.sum_exp,
                };
            }
            Less => [a, b],
            Greater => [b, a],
        };
        Self {
            max: g.max,
            sum_exp: l.sum_exp * (l.max - g.max).exp() + g.sum_exp,
        }
    }
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use std::iter::zip;

    #[test]
    fn test_online_softmax() {
        let data: Vec<f64> = (0..4096)
            .map(|_| (rand::random::<f64>() - 0.5) * 20.)
            .collect();

        // 计算标准 softmax
        let mut ans = data.clone();
        safe_softmax(&mut ans);

        // 计算 online softmax
        let mut data = data;
        online_softmax(&mut data, 32);

        for (ans, res) in zip(ans, data) {
            assert!((ans - res).abs() < f64::EPSILON)
        }
    }

    /// 标准 softmax 实现
    pub fn safe_softmax(data: &mut [f64]) {
        // 找到最大值以提高数值稳定性
        let mut max = f64::NEG_INFINITY;
        for &x in &*data {
            max = f64::max(max, x)
        }

        // 计算指数并求和
        let mut sum = 0.;
        for x in &mut *data {
            *x = (*x - max).exp();
            sum += *x
        }

        // 归一化
        for x in data {
            *x /= sum
        }
    }
}
