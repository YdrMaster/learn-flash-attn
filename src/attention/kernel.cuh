#include <cub/warp/warp_reduce.cuh>
#include <cuda/std/cstddef>

template <typename T>
__host__ __device__ T neg_inf();

template <>
__host__ __device__ float neg_inf<float>() {
    return __int_as_float(0xFF800000);
}

template <>
__host__ __device__ double neg_inf<double>() {
    return __longlong_as_double(0xFFF0000000000000ULL);
}

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
    Strides2D kv_strides;
    // output
    T *o;
    Strides2D o_strides;
    // config
    bool *const mask;
    // 用于表明是否为causal mask，如果是则忽略mask指针，在计算时自动生成
    bool is_causal_mask;
    size_t n, s;
};

// threads (b) (kvh, d)
template <typename T>
__device__ void cache_concat_block(
    KernelCfg cfg,
    KVPage<T> const *cache_pages,
    KernelReq<T> const *reqs) {
    size_t const
        ireq = blockIdx.x,
        head = threadIdx.y,
        it = threadIdx.x;

    KernelReq const req = reqs[ireq];
    KVPage<T> const *pages = cache_pages + req.pages_start;
    size_t const
        bs = cfg.bs,
        pos = req.s - req.n;
    for (size_t i = 0; i < req.n; ++i) {
        KVPage const page = pages[(pos + i) / bs];
        ptrdiff_t const
            k_offset = req.k_strides.offset(head, i),
            v_offset = req.v_strides.offset(head, i),
            c_offset = req.kv_strides.offset(head, (pos + i) % bs);
        byte_offset(page.k, c_offset)[it] = byte_offset(req.k, k_offset)[it];
        byte_offset(page.v, c_offset)[it] = byte_offset(req.v, v_offset)[it];
    }
}

// threads (b, h) (bn, warp)
template <typename Tcompute, typename T>
__device__ void flash_attn_block(
    KernelCfg cfg,
    KVPage<T> const *cache_pages,
    KernelReq<T> const *reqs) {
    size_t const
        ireq = blockIdx.y,
        head = blockIdx.x,
        bn = blockDim.y,
        warp = blockDim.x,
        it_y = threadIdx.y,
        lane = threadIdx.x;

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
      *oi = qi + bn * d,
      *kj = oi + bn * d,
      *vj = kj + bs * d,
      *x = vj + bs * d;
    // 定位每个线程的 qi, oi, x
    qi += it_y * d;
    oi += it_y * d;
    x += it_y * bs;

    size_t const
        ikvb_end = div_ceil(req.s, bs),
        iqb_end = div_ceil(req.n, bn);
    // seq 方向迭代
    for (size_t iqb = 0; iqb < iqb_end; ++iqb) {
        size_t const iq = iqb * bn + it_y;
        T const *req_q = byte_offset(req.q, req.q_strides.offset(head, iq));
        T /***/ *req_o = byte_offset(req.o, req.o_strides.offset(head, iq));
        if (iq < req.n) {
            for (size_t i = lane; i < d; i += warp) {
                qi[i] = req_q[i]; // 加载 qi
                oi[i] = 0;        // 初始化 oi 为 0
            }
        }
        // 初始化 m l
        Tcompute mi_1 = neg_inf<Tcompute>(),
                 di_1 = 0;
        // ctx 方向迭代
        for (size_t ikvb = 0; ikvb < ikvb_end; ++ikvb) {
            bool const *mask = req.mask + (iq * ikvb_end + ikvb) * bs;
            // 每个线程拷贝 k/v 的一行，拷贝整个 kv block 到 local memory
            T const
                *k = (cache_pages + req.pages_start + ikvb)->k,
                *v = (cache_pages + req.pages_start + ikvb)->v;
            for (size_t i = it_y; i < bs; i += bn) {
                ptrdiff_t const offset = req.kv_strides.offset(head / g, i);
                for (size_t j = lane; j < d; j += warp) {
                    kj[i * d + j] = byte_offset(k, offset)[j];
                    vj[i * d + j] = byte_offset(v, offset)[j];
                }
            }
            __syncthreads();
            // 每个线程束计算 q 的一行
            if (iq < req.n) {
                Tcompute mi = mi_1,
                         sum = 0;

                // 用于计算 mask
                size_t pos = req.s - req.n;
                size_t kv_pos_base = ikvb * bs;

                // score = q @ k^T / √d
                for (size_t i = 0; i < bs; ++i) {
                    bool is_valid;
                    if (req.is_causal_mask) {
                        size_t kv_pos = kv_pos_base + i;
                        is_valid = kv_pos <= pos + iq;
                    } else {
                        is_valid = mask[i];
                    }

                    if (!is_valid) {
                        continue;
                    }
                    Tcompute qk = 0;
                    for (size_t j = lane; j < d; j += warp) {
                        qk += (Tcompute)(qi[j] * kj[i * d + j]);
                    }
                    {
                        using WarpReduce = cub::WarpReduce<Tcompute>;
                        __shared__ typename WarpReduce::TempStorage temp_storage;
                        qk = __shfl_sync(0xFFFFFFFF, WarpReduce(temp_storage).Sum(qk, d), 0);
                        __syncwarp();
                    }
                    qk *= scale;
                    // lane 0
                    if (lane == 0) {
                        x[i] = qk;
                    }
                    if (mi < qk) {
                        mi = qk;
                    }
                }
                for (size_t i = 0; i < bs; ++i) {
                    bool is_valid;
                    if (req.is_causal_mask) {
                        size_t kv_pos = kv_pos_base + i;
                        is_valid = kv_pos <= pos + iq;
                    } else {
                        is_valid = mask[i];
                    }

                    if (!is_valid) {
                        if (lane == 0) {
                            x[i] = 0;
                        }
                    } else {
                        Tcompute exp = ::exp((Tcompute)x[i] - mi);
                        sum += exp;
                        if (lane == 0) {
                            x[i] = exp;
                        }
                    }
                }

                Tcompute exp = ::exp(mi_1 - mi),
                         exp_mut_di_1 = di_1 * exp,
                         di = exp_mut_di_1 + sum;
                // 更新 m, l
                mi_1 = mi;
                di_1 = di;
                // 同步 m, d
                for (size_t i = lane; i < d; i += warp) {
                    Tcompute xv = 0;
                    for (size_t k = 0; k < bs; ++k) {
                        xv += (Tcompute)(x[k] * vj[k * d + i]);
                    }
                    oi[i] = (Tcompute)oi[i] * exp + xv;
                }
            }
            __syncthreads();
        }
        // 将 oi 写入 o
        if (iq < req.n) {
            for (size_t i = lane; i < d; i += warp) {
                req_o[i] = (Tcompute)oi[i] / di_1;
            }
        }
    }
}
