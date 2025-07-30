#include <cub/warp/warp_reduce.cuh>
#include <cuda/std/cstddef>

// 生成平台无关的负无穷编码

template <typename T>
__host__ __device__ __forceinline__ T neg_inf();

template <>
__host__ __device__ __forceinline__ float neg_inf<float>() {
    return __int_as_float(0xFF800000);
}

template <>
__host__ __device__ __forceinline__ double neg_inf<double>() {
    return __longlong_as_double(0xFFF0000000000000ULL);
}

// 调用特定类型的融合乘加命令

template <typename T>
__host__ __device__ __forceinline__ void fma_(T const &a, T const &b, T &c) {
    c += a * b;
}

template <>
__host__ __device__ __forceinline__ void fma_<float>(float const &a, float const &b, float &c) {
    c = fmaf(a, b, c);
}

// 工具函数

__host__ __device__ __forceinline__ size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}

template <typename T>
__host__ __device__ __forceinline__ T *byte_offset(T *ptr, ptrdiff_t diff) {
    return (T *)(((char *)ptr) + diff);
}

// 参数数据类型

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

    __host__ __device__ __forceinline__ ptrdiff_t offset(size_t head_, size_t seq_) const {
        return head_ * head + seq_ * seq;
    }
};

enum class AttnType : uint8_t {
    Full = 0,
    Causal = 1,
    CustomMask = 255,
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
    size_t n, s;
    AttnType ty;
    bool *const mask;
};

// 连接 kv cache

// threads (b, kvh) (d)
template <typename T>
__device__ void cache_concat_block(
    KernelCfg cfg,
    KVPage<T> const *cache_pages,
    KernelReq<T> const *reqs) {
    size_t const
        ireq = blockIdx.y,
        head = blockIdx.x,
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

// flash attention 本体

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
    Tcompute const
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
        size_t const
            iq = iqb * bn + it_y,
            causal_end = req.ty == AttnType::Causal ? req.s - req.n + iq : (size_t)-1;
        T const *req_q = byte_offset(req.q, req.q_strides.offset(head, iq));
        T /***/ *req_o = byte_offset(req.o, req.o_strides.offset(head, iq));
        if (iq < req.n) {
            load_qo(qi, oi, req_q);
        }
        // 初始化 m l
        Tcompute mi_1 = neg_inf<Tcompute>(),
                 di_1 = 1e-6; // 避免除 0
        // ctx 方向迭代
        for (size_t ikvb = 0; ikvb < ikvb_end; ++ikvb) {
            size_t const kv_base = ikvb * bs;
            if (kv_base > causal_end) {
                break;
            }
            size_t const s_end =
                kv_base + bs > causal_end
                    ? causal_end - kv_base + 1
                    : bs;

            // 每个线程拷贝 k/v 的一行，拷贝整个 kv block 到 local memory
            T const *k = (cache_pages + req.pages_start + ikvb)->k,
                    *v = (cache_pages + req.pages_start + ikvb)->v;
            for (size_t i = it_y; i < bs; i += bn) {
                ptrdiff_t const offset = req.kv_strides.offset(head / g, i);
                load_kv(kj + i * d, vj + i * d, byte_offset(k, offset), byte_offset(v, offset));
            }
            __syncthreads();
            // 每个线程束计算 q 的一行
            if (iq < req.n) {
                bool const *mask = req.mask + iq * ikvb_end * bs + kv_base;
                Tcompute mi = mi_1, sum = 0;
                // score = q @ k^T / √d
                for (size_t i = 0; i < s_end; ++i) {
                    if (req.ty == AttnType::CustomMask && !mask[i]) {
                        continue;
                    }

                    Tcompute qk = 0;
                    for (size_t j = lane; j < d; j += warp) {
                        fma_<Tcompute>(qi[j], kj[i * d + j], qk);
                    }
                    {
                        using WarpReduce = cub::WarpReduce<Tcompute>;
                        __shared__ typename WarpReduce::TempStorage temp_storage;
                        qk = __shfl_sync(0xFFFFFFFF, WarpReduce(temp_storage).Sum(qk, d), 0);
                        __syncwarp();
                    }
                    qk *= scale;
                    if (lane == 0) {
                        x[i] = qk;
                    }
                    if (mi < qk) {
                        mi = qk;
                    }
                }
                for (size_t i = 0; i < s_end; ++i) {
                    if (req.ty == AttnType::CustomMask && !mask[i]) {
                        continue;
                    }

                    Tcompute exp = ::exp((Tcompute)x[i] - mi);
                    sum += exp;
                    if (lane == 0) {
                        x[i] = exp;
                    }
                }

                Tcompute const exp = ::exp(mi_1 - mi);
                // 更新 m, l
                mi_1 = mi;
                di_1 = di_1 * exp + sum;
                for (size_t i = lane; i < d; i += warp) {
                    Tcompute xv = 0;
                    for (size_t j = 0; j < s_end; ++j) {
                        if (req.ty == AttnType::CustomMask && !mask[j]) {
                            continue;
                        }
                        fma_<Tcompute>(x[j], vj[j * d + i], xv);
                    }
                    oi[i] = (Tcompute)oi[i] * exp + xv;
                }
            }
            __syncthreads();
        }
        // 将 oi 写入 o
        if (iq < req.n) {
            Tcompute k = 1 / di_1;
            for (size_t i = lane; i < d; i += warp) {
                req_o[i] = (Tcompute)oi[i] * k;
            }
        }
    }
}
