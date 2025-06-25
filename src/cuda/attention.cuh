#include <cstdint>
#include <math_constants.h>

template <typename T>
struct kv_cache {
    T *k, *v;
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
    T *const *kv_pages,
    T const *q_,           // n@ x d
    T *o_,                 // n@ x d
    bool const *mask_,     // n x s
    T *m,                  // s
    T *l,                  // s
    T *qi_,                // bn x d
    T *kj,                 // bs x d
    T *vj,                 // bs x d
    T *x_,                 // bn x bs
    uint64_t const n,      // sequence length
    uint64_t const d,      // head dim
    uint64_t const ts,     // = s/bs
    uint64_t const bs,     // context tile
    int64_t const sq,      //        q stride
    int64_t const so,      //        o stride
    int64_t const kv_sbuf, // sequence stride
    int64_t const kv_skv,  //   k to v stride
    int64_t const kv_sh,   //  kv head stride
    float const scale) {
    // (batch x head) x (bn)
    uint64_t const head = blockIdx.x;
    uint64_t const bn = blockDim.x;
    uint64_t const it = threadIdx.x;
    uint64_t const tn = (n + bn - 1) / bn;
    // kv
    for (uint64_t ikvb = 0; ikvb < ts; ++ikvb) {
        { // 每个线程拷贝 k/v 的一行，拷贝整个 kv block 到 local memory
            uint64_t const end = (ikvb + 1) * bs;
            for (uint64_t ikv = ikvb * bs + it, i = it; ikv < end; ikv += bn, i += bn) {
                kv_cache const cache = locate_cache(kv_pages, kv_sbuf, kv_skv, kv_sh, bs, head, ikv);
                for (uint64_t j = 0; j < d; ++j) {
                    kj[i * d + j] = k[j];
                    vj[i * d + j] = v[j];
                }
            }
            __syncthreads();
        }
        { // 每个线程计算 q 的一行
            T *qi = qi_ + it * d;
            T *x = x_ + it * bs;

            for (uint64_t iqb = 0; iqb < tn; ++iqb) {
                uint64_t iq = iqb * bn + it;
                if (iq >= n) {
                    break;
                }
                // locate data
                T const *q = q_ + iq * sq;
                T *o = o_ + iq * so;
                bool const *mask = mask + iq * s + ikvb * bs;
                // load data
                for (uint64_t i = 0; i < d; ++i) {
                    qi[i] = q[i];
                }

                T const mi_1 = m[iq];
                T const di_1 = l[iq];

                // score = q @ k^T / √d
                T mi = mi_1;
                for (uint64_t i = 0; i < bs; ++i) {
                    if (!mask[i]) {
                        x[i] = -CUDART_INF_F;
                    } else {
                        T const *k = kj + i * d;

                        for (uint64_t j = 0; j < d; ++j) {
                            x[i] += qi[j] * kj[j];
                        }
                        x[i] *= scale;

                        if (x[i] > mi) {
                            mi = x[i];
                        }
                    }
                }

                T sum = 0;
                for (uint64_t i = 0; i < bs; ++i) {
                    x[i] = std::exp(x[i] - mi);
                    sum += x[i];
                }

                T exp = di_1 * std::exp(mi_1 - mi);
                T di = exp + sum;

                m[iq] = mi;
                l[iq] = di;

                T rdi = 1 / di;
                exp *= rdi;
                for (uint64_t i = 0; i < bs; ++i) {
                    x[i] *= rdi;
                }
            }
            __syncthreads();
        }
    }
}
