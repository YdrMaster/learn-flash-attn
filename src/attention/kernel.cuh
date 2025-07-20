#include <cuda/std/cstddef>

__device__ size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ T *byte_offset(T *ptr, ptrdiff_t diff) {
    return (T *)(((char *)ptr) + diff);
}

struct KernelCfg {
    size_t g, d, bs;
    float scale;
};

template <typename T>
struct KVPage {
    T *k, *v;
};

struct Strides2D {
    ptrdiff_t head, seq;

    __device__ ptrdiff_t offset(size_t head_, size_t seq_) const {
        return head_ * head + seq_ * seq;
    }
};

template <typename T>
struct KernelReq {
    // qurey
    T const *q;
    Strides2D q_strides;
    T const *k;
    Strides2D k_strides;
    T const *v;
    Strides2D v_strides;
    // kv (paged)
    size_t pages_start;
    T *kv_cache; //[buf, 2, kvh, d]
    Strides2D kv_strides;
    // output
    T *o;
    Strides2D o_strides;
    // config
    bool *const mask;
    T *l, *m;
    size_t n, s;
};

template <typename Tcompute, typename T>
__device__ void flash_attn_block(
    KernelCfg cfg,
    KVPage<T> const *cache_pages,
    KernelReq<T> const *reqs) {
    size_t const
        ireq = blockIdx.y,
        head = blockIdx.x,
        bn = blockDim.x,
        it = threadIdx.x;

    size_t const
        g = cfg.g,
        d = cfg.d,
        bs = cfg.bs;
    float const
        scale = cfg.scale;
    KernelReq const
        req = reqs[ireq];
    // 划分 shared memory
    extern __shared__ T shared[];
    T *qi = shared,
      *kj = qi + bn * d,
      *vj = kj + bs * d,
      *x = vj + bs * d;
    // 定位每个线程的 qi, x
    qi += it * d;
    x += it * bs;

    size_t const
        ikvb_end = div_ceil(req.s, bs),
        iqb_end = div_ceil(req.n, bn);

    // ctx 方向迭代
    for (size_t ikvb = 0; ikvb < ikvb_end; ++ikvb) {
        // 每个线程拷贝 k/v 的一行，拷贝整个 kv block 到 local memory
        {
            T const
                *k = (cache_pages + req.pages_start + ikvb)->k,
                *v = (cache_pages + req.pages_start + ikvb)->v;
            for (size_t i = it; i < bs; i += bn) {
                ptrdiff_t const offset = req.kv_strides.offset(head / g, i);
                memcpy(kj + i * d, byte_offset(k, offset), d * sizeof(T));
                memcpy(vj + i * d, byte_offset(v, offset), d * sizeof(T));
            }
        }
        __syncthreads();
        // seq 方向迭代
        for (size_t iqb = 0; iqb < iqb_end; ++iqb) {
            // 每个线程计算 q 的一行
            size_t iq = iqb * bn + it;
            if (iq >= req.n) {
                continue;
            }
            // locate data
            T const *q_ = byte_offset(req.q, req.q_strides.offset(head, iq));
            T /***/ *o_ = byte_offset(req.o, req.o_strides.offset(head, iq));
            // load data
            memcpy(qi, q_, d * sizeof(T));
            bool const *mask = req.mask + iq * req.s + ikvb * bs;

            T *m = req.m + head * req.s + iq,
              *l = req.l + head * req.s + iq;
            Tcompute const mi_1 = *m,
                           di_1 = *l;

            // score = q @ k^T / √d
            Tcompute mi = mi_1;
            for (size_t i = 0; i < bs; ++i) {
                if (mask[i]) {
                    Tcompute qk = 0;
                    for (size_t j = 0; j < d; ++j) {
                        qk += qi[j] * kj[i * d + j];
                    }
                    qk *= scale;
                    x[i] = qk;
                    if (qk > mi) {
                        mi = qk;
                    }
                }
            }

            Tcompute sum = 0;
            for (size_t i = 0; i < bs; ++i) {
                if (!mask[i]) {
                    x[i] = 0;
                } else {
                    x[i] = ::exp(x[i] - mi);
                    sum += x[i];
                }
            }

            Tcompute exp = di_1 * ::exp(mi_1 - mi),
                     di = exp + sum;
            // 更新 m, l
            *m = mi;
            *l = di;

            for (size_t j = 0; j < d; ++j) {
                Tcompute xv = 0;
                for (size_t i = 0; i < bs; ++i) {
                    xv += x[i] * vj[i * d + j];
                }
                o_[j] = (o_[j] * exp + xv) / di;
            }
        }
    }
}
// shape (b) (kvh,bn)
template <typename Tcompute, typename T>
__device__ void cache_concat_kernel(
    KernelCfg cfg,
    KernelReq<T> const *reqs) {
    size_t const
        ireq = blockIdx.x,
        bn = blockDim.x,
        ib = threadIdx.x,
        kvh = blockDim.y,
        ikv = threadIdx.y;

    KernelReq const req = reqs[ireq];
    size_t const
        d = cfg.d;

    // 计算每个线程处理的token数量
    size_t tokens_per_thread = div_ceil(req.n, bn);

    size_t global_idx = req.s - req.n,
           start_idx = ib * tokens_per_thread;
    size_t d_size = d * sizeof(T);

    for (size_t i = 0; i < tokens_per_thread; i++) {
        size_t token_idx = start_idx + i;
        if (token_idx >= req.n) {
            break;
        }
        // 计算在kv缓存中的绝对位置
        global_idx += token_idx;

        ptrdiff_t c_offset = req.kv_strides.offset(ikv, global_idx);
        ptrdiff_t k_offset = req.k_strides.offset(ikv, token_idx);
        ptrdiff_t v_offset = req.v_strides.offset(ikv, token_idx);
        ptrdiff_t v_cache_offset = c_offset + req.kv_strides.head * kvh;

        memcpy(byte_offset(req.kv_cache, c_offset),
               byte_offset(req.k, k_offset),
               d_size);
        memcpy(byte_offset(req.kv_cache, v_cache_offset),
               byte_offset(req.v, v_offset),
               d_size);
    }
}
