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

// threads (b, kvh.h) (kvh.l, warp)
template <typename T>
__device__ void cache_concat_block(
    KernelCfg cfg,
    KVPage<T> const *cache_pages,
    KernelReq<T> const *reqs) {
    size_t const
        ireq = blockIdx.y,
        // nh_h = gridDim.x,
        ih_h = blockIdx.x,
        nh_l = blockDim.y,
        ih_l = threadIdx.y,
        head = ih_h * nh_l + ih_l,
        warp = blockDim.x,
        lane = threadIdx.x;

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
        load_kv(byte_offset(page.k, c_offset),
                byte_offset(page.v, c_offset),
                byte_offset(req.k, k_offset),
                byte_offset(req.v, v_offset));
    }
}

// flash attention 本体

// threads (b, h, bn) (bs, warp)
template <typename Tcompute, typename T>
__device__ void flash_attn_block(
    KernelCfg cfg,
    KVPage<T> const *cache_pages,
    KernelReq<T> const *reqs) {
    size_t const
        ireq = blockIdx.z,
        head = blockIdx.y,
        q_base = blockIdx.x,
        bn = gridDim.x,
        i_warp = threadIdx.y,
        bs = blockDim.y,
        lane = threadIdx.x,
        warp = blockDim.x;

    size_t const
        g = cfg.g,
        d = cfg.d;
    Tcompute const
        scale = cfg.scale;
    KernelReq const
        req = reqs[ireq];
    // 划分 shared memory
    extern __shared__ T shared[];
    T *qi = shared,
      *oi = qi + d,
      *kj = oi + d,
      *vj = kj + bs * d,
      *x = vj + bs * d;

    size_t const
        ikvb_end = div_ceil(req.s, bs),
        iqb_end = div_ceil(req.n, bn);
    // seq 方向迭代
    for (size_t iq = q_base; iq < req.n; iq += bn) {
        size_t const causal_end = req.ty == AttnType::Causal ? req.s - req.n + iq : (size_t)-1;
        T const *req_q = byte_offset(req.q, req.q_strides.offset(head, iq));
        T /***/ *req_o = byte_offset(req.o, req.o_strides.offset(head, iq));
        for (size_t i = i_warp * warp + lane; i < d; i += bs * warp) {
            qi[i] = req_q[i];
            oi[i] = 0;
        }
        __syncthreads();
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
            ptrdiff_t const offset = req.kv_strides.offset(head / g, i_warp);
            load_kv(kj + i_warp * d, vj + i_warp * d, byte_offset(k, offset), byte_offset(v, offset));
            __syncthreads();
            //
            bool const *mask = req.mask + iq * ikvb_end * bs + kv_base;
            Tcompute mi = mi_1, sum = 0;
            // score = q @ k^T / √d
            {
                Tcompute qk = 0;
                for (size_t j = lane; j < d; j += warp) {
                    fma_<Tcompute>(qi[j], kj[i_warp * d + j], qk);
                }
                using WarpReduce = cub::WarpReduce<Tcompute>;
                __shared__ typename WarpReduce::TempStorage temp_storage;
                qk = WarpReduce(temp_storage).Sum(qk, d);
                if (lane == 0) {
                    x[i_warp] = qk * scale;
                }
            }
            __syncthreads();
            // max
            for (size_t i = 0; i < s_end; ++i) {
                if (mi < (Tcompute)x[i]) {
                    mi = (Tcompute)x[i];
                }
            }
            // score = exp(score - max)
            if (lane == 0) {
                x[i_warp] = ::exp((Tcompute)x[i_warp] - mi);
            }
            __syncthreads();
            // sum
            for (size_t i = 0; i < s_end; ++i) {
                if (req.ty == AttnType::CustomMask && !mask[i]) {
                    continue;
                }
                sum += (Tcompute)x[i];
            }
            Tcompute const exp = ::exp(mi_1 - mi);
            // 更新 m, l
            mi_1 = mi;
            di_1 = di_1 * exp + sum;
            // score = score * v
            for (size_t i = lane; i < d; i += warp) {
                Tcompute xv = 0;
                for (size_t j = 0; j < bs; ++j) {
                    if (j >= s_end || (req.ty == AttnType::CustomMask && !mask[i])) {
                        continue;
                    }
                    fma_<Tcompute>(x[j], vj[j * d + i], xv);
                }
                if (i_warp == 0) {
                    oi[i] = (Tcompute)oi[i] * exp + xv;
                }
            }
            __syncthreads();
        }
        // 将 oi 写入 o
        Tcompute k = 1 / di_1;
        for (size_t i = i_warp * warp + lane; i < d; i += bs * warp) {
            req_o[i] = (Tcompute)oi[i] * k;
        }
    }
}
