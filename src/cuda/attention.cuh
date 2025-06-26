#include <cstdio>
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
    T *const *kv_pages,
    T const *q_,           // n@ x d
    T *o_,                 // n@ x d
    bool const *mask_,     // n x s
    T *m,                  // s
    T *l,                  // s
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

    extern __shared__ T sram[];
    int tile_size = bs * d;
    T *qi = sram;
    T *kj = &sram[tile_size];
    T *vj = &sram[tile_size * 2];
    T *x = new T[bs];
    // kv
    for (uint64_t ikvb = 0; ikvb < ts; ++ikvb) {
        // 加载kv
        { // 每个线程拷贝 k/v 的一行，拷贝整个 kv block 到 local memory
            uint64_t const end = (ikvb + 1) * bs;
            for (uint64_t ikv = ikvb * bs + it, i = it; ikv < end; ikv += bn, i += bn) {
                kv_cache const cache = locate_cache(kv_pages, kv_sbuf, kv_skv, kv_sh, bs, head, ikv);
                for (uint64_t j = 0; j < d; ++j) {
                    kj[i * d + j] = cache.k[j];
                    vj[i * d + j] = cache.v[j];
                }
            }
            __syncthreads();
        }
        { // 每个线程计算 q 的一行

            for (uint64_t iqb = 0; iqb < tn; ++iqb) {
                uint64_t iq = iqb * bn + it;
                if (iq >= n) {
                    break;
                }
                // locate data 加载q
                T const *q = q_ + iq * sq;
                T *o = o_ + iq * so;
                bool const *mask = mask_ + iq * n + ikvb * bs;
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
                            x[i] += qi[j] * k[j];
                        }
                        x[i] *= scale;

                        if (x[i] > mi) {
                            mi = x[i];
                        }
                    }
                }
                // P = exp(S - row_m), row_l = rowsum(P)
                T sum = 0;
                for (uint64_t i = 0; i < d; ++i) {
                    x[i] = ::exp(x[i] - mi);
                    sum += x[i];
                }

                T exp = di_1 * ::exp(mi_1 - mi);
                T di = exp + sum;
                // 更新mi,di
                m[iq] = mi;
                l[iq] = di;

                T rdi = 1 / di;
                exp *= rdi;
                for (uint64_t i = 0; i < bs; ++i) {
                    x[i] *= rdi;
                }
                T *xv = new T[d];
                for (uint64_t i = 0; i < d; ++i) {
                    xv[i] = 0;
                }
                for (uint64_t i = 0; i < bs; ++i) {
                    T xi = x[i];
                    T *vji = &vj[i * d];
                    for (uint64_t j = 0; j < d; ++j) {
                        xv[j] += xi * vji[j];
                    }
                }
                for (uint64_t j = 0; j < d; ++j) {
                    o[j] = o[j] * exp + xv[j];
                }
            }
            __syncthreads();
        }
    }
}
extern "C" __global__ void __attn_f64(
    double *const *kv_pages,
    double const *q_,
    double *o_,
    bool const *mask_,
    double *m,
    double *l,
    uint64_t const n,
    uint64_t const d,
    uint64_t const ts,
    uint64_t const bs,
    int64_t const sq,
    int64_t const so,
    int64_t const kv_sbuf,
    int64_t const kv_skv,
    int64_t const kv_sh,
    float const scale) {
    // 打印所有参数
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("kv_pages: %p\n", kv_pages);
        for (uint64_t i = 0; i < ts; ++i) {
            printf("  kv_pages[%lu]: %p\n", i, kv_pages[i]);
            double *page = kv_pages[i];
            for (uint64_t j = 0; j < bs * d * 2; ++j) {
                printf("    kv_pages[%lu][%lu]=%f\n", i, j, page[j]);
            }
        }
        printf("q_: %p\n", q_);
        for (uint64_t i = 0; i < n * d; ++i) {
            printf("  q_[%lu]=%f\n", i, q_[i]);
        }
        printf("o_: %p\n", o_);
        for (uint64_t i = 0; i < n * d; ++i) {
            printf("  o_[%lu]=%f\n", i, o_[i]);
        }
        printf("mask_: %p\n", (void *)mask_);
        for (uint64_t i = 0; i < n * n; ++i) {
            printf("  mask_[%lu] = %s\n", i, mask_[i] ? "true" : "false");
        }
        printf("m: %p\n", m);
        for (uint64_t i = 0; i < n; ++i) {
            printf("  m[%lu]=%f\n", i, m[i]);
        }
        printf("l: %p\n", l);
        for (uint64_t i = 0; i < n; ++i) {
            printf("  l[%lu]=%f\n", i, l[i]);
        }
        printf("n: %lu\n", n);
        printf("d: %lu\n", d);
        printf("ts: %lu\n", ts);
        printf("bs: %lu\n", bs);
        printf("sq: %ld\n", sq);
        printf("so: %ld\n", so);
        printf("kv_sbuf: %ld\n", kv_sbuf);
        printf("kv_skv: %ld\n", kv_skv);
        printf("kv_sh: %ld\n", kv_sh);
        printf("scale: %f\n", scale);
    }
    // 调用模板实现
    __attn<double>(kv_pages, q_, o_, mask_, m, l, n, d, ts, bs, sq, so, kv_sbuf, kv_skv, kv_sh, scale);
}
