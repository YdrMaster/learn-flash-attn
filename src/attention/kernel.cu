#include <cuda/std/cstdint>
#include <math_constants.h>

template <typename T>
struct kv_cache {
    T *k;
    T *v;
    __device__ kv_cache(T *k_, T *v_) : k(k_), v(v_) {}
};

template <typename T>
__device__ kv_cache<T> locate_cache(
    T *const *pages,
    int64_t sbuf,      // sequence stride
    int64_t skv,       //   k to v stride
    int64_t sh,        //  kv head stride
    uint64_t const bs, // context tile
    uint64_t const head,
    uint64_t const pos) {
    sh *= head;
    sbuf *= pos % bs;
    uint8_t *page = (uint8_t *)pages[pos / bs];
    return kv_cache{
        (T *)(page + sh + sbuf),
        (T *)(page + sh + sbuf + skv),
    };
}
template <typename T>
__device__ void __attn(
    T *const **kv_pages,    // [batch][page]
    T *const *q_,           // batch [n@ x d]
    T **o_,                 // batch [n@ x d]
    bool *const *mask_,     // batch [n x s]
    T **m,                  // batch [s]
    T **l,                  // batch [s]
    uint64_t const *n,      // batch [sequence length]
    uint64_t const d,       // head dim
    uint64_t const *ts,     // batch [s/bs]
    uint64_t const bs,      // context tile
    uint64_t const g,       // GQA
    int64_t const sq,       // q stride
    int64_t const so,       // o stride
    int64_t const *kv_sbuf, // batch [sequence stride]
    int64_t const *kv_skv,  // batch [k to v stride]
    int64_t const *kv_sh,   // batch [kv head stride]
    float const scale) {
    // (batch x head) x (bn)
    uint64_t const b = blockIdx.y;
    uint64_t const head = blockIdx.x;
    uint64_t const bn = blockDim.x;
    uint64_t const it = threadIdx.x;
    uint64_t const tn = (n[b] + bn - 1) / bn;

    // 当前批次参数
    uint64_t const n_b = n[b];
    uint64_t const ts_b = ts[b];
    int64_t const kv_sbuf_b = kv_sbuf[b];
    int64_t const kv_skv_b = kv_skv[b];
    int64_t const kv_sh_b = kv_sh[b];

    // 目前把所有维度当作连续的
    extern __shared__ T sram[];
    T *kj = sram;
    T *vj = sram + bs * d;

    // 当前批次指针
    T const *q_batch = q_[b];
    T *o_batch = o_[b];
    bool const *mask_batch = mask_[b];
    T *m_batch = m[b];
    T *l_batch = l[b];
    T *const *pages_batch = kv_pages[b];

    // 当前头在Q/K/V中的偏移
    T const *q = q_batch + head * n_b * d;
    T *o = o_batch + head * n_b * d;
    for (uint64_t iqb = 0; iqb < tn; ++iqb) {
        uint64_t iq = iqb * bn + it;
        if (iq >= n_b) {
            continue;
        }

        // 加载当前查询位置的状态
        T mi = m_batch[head * n_b + iq];
        T li = l_batch[head * n_b + iq];
        T *oi = o + iq * d;

        for (uint64_t ikvb = 0; ikvb < ts_b; ++ikvb) {
            // 加载 kv block 到共享内存
            {
                uint64_t const end = (ikvb + 1) * bs;
                for (uint64_t ikv = ikvb * bs + it, i = it; ikv < end; ikv += bn, i += bn) {
                    kv_cache<T> cache = locate_cache<T>(pages_batch, kv_sbuf_b, kv_skv_b, kv_sh_b, bs, head / g, ikv);
                    for (uint64_t j = 0; j < d; ++j) {
                        kj[i * d + j] = cache.k[j];
                        vj[i * d + j] = cache.v[j];
                    }
                }
                __syncthreads();
            }
            // 加载 q_i 和 mask
           T *qi_val = new T[d];
            bool const *mask = mask_batch + iq * ts_b * bs + ikvb * bs;

            for (uint64_t j = 0; j < d; ++j) {
                qi_val[j] = q[iq * d + j];
            }

            // 计算注意力分数
            T *scores = new T[bs];
            T mi_local = -CUDART_INF_F;
            for (uint64_t i = 0; i < bs; ++i) {
                if (!mask[i]) {
                    scores[i] = -CUDART_INF_F;
                } else {
                    scores[i] = 0;
                    for (uint64_t j = 0; j < d; ++j) {
                        scores[i] += qi_val[j] * kj[i * d + j];
                    }
                    scores[i] *= scale;
                    if (scores[i] > mi_local) {
                        mi_local = scores[i];
                    }
                }
            }

            T mi_new = max(mi, mi_local);
            T sum = 0.0;

            for (uint64_t i = 0; i < bs; ++i) {
                if (mask[i]) {
                    scores[i] = exp(scores[i] - mi_new);
                    sum += scores[i];
                } else {
                    scores[i] = 0;
                }
            }

            T exp_old = (mi == -CUDART_INF_F) ? 0.0 : exp(mi - mi_new);
            T li_new = exp_old * li + sum;
            T rdi = (li_new == 0) ? 0.0 : 1.0 / li_new;

            // 更新输出
            for (uint64_t j = 0; j < d; ++j) {
                T v_acc = 0.0;
                for (uint64_t i = 0; i < bs; ++i) {
                    if (mask[i]) {
                        v_acc += scores[i] * vj[i * d + j];
                    }
                }
                oi[j] = oi[j] * exp_old * li * rdi + v_acc * rdi;
            }
            mi = mi_new;
            li = li_new;
        }
        // 保存最终状态
        m_batch[head * n_b + iq] = mi;
        l_batch[head * n_b + iq] = li;
    }
}

extern "C" __global__ void __attn_f64(
    double *const **kv_pages,
    double *const *q_,
    double **o_,
    bool *const *mask_,
    double **m,
    double **l,
    uint64_t const *n,
    uint64_t const d,
    uint64_t const *ts,
    uint64_t const bs,
    uint64_t const g,
    int64_t const sq,
    int64_t const so,
    int64_t const *kv_sbuf,
    int64_t const *kv_skv,
    int64_t const *kv_sh,
    float const scale) {
    __attn<double>(
        kv_pages, q_, o_, mask_, m, l,
        n, d, ts, bs, g, sq, so,
        kv_sbuf, kv_skv, kv_sh, scale);
}
