//! READ: <https://zhuanlan.zhihu.com/p/11656282335>

/// 增量 softmax 实现
pub fn online_softmax(data: &mut [f64]) {
    // 一次遍历，同时确定最大值和指数和
    let mut s = S::EMPTY;
    for &x in &*data {
        s.update(x)
    }

    let S {
        current_max: max,
        sum_exp: sum,
    } = s;

    // 归一化
    for x in data {
        *x = (*x - max).exp() / sum
    }
}

struct S {
    current_max: f64,
    sum_exp: f64,
}

impl S {
    const EMPTY: Self = Self {
        current_max: f64::NEG_INFINITY,
        sum_exp: 0.,
    };

    /// 根据新元素更新 softmax 状态
    fn update(&mut self, x: f64) {
        self.sum_exp = if x > self.current_max {
            let old_max = std::mem::replace(&mut self.current_max, x);
            self.sum_exp * (old_max - x).exp() + 1.
        } else {
            self.sum_exp + (x - self.current_max).exp()
        }
    }
}

#[cfg(test)]
mod tests {
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
        online_softmax(&mut data);

        for (ans, res) in zip(ans, data) {
            assert!((ans - res).abs() < f64::EPSILON)
        }
    }

    /// 标准 softmax 实现
    fn safe_softmax(data: &mut [f64]) {
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
