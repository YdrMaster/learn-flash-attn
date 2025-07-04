#include <cuda/std/cstddef>

using Tdata = double;

__device__ size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ T *byte_offset(T *ptr, ptrdiff_t diff) {
    return (T *)(((char *)ptr) + diff);
}

struct KernelCfg {
    size_t g, d, bs;
    Tdata scale;
};

struct KVPage {
    Tdata *k, *v;
};

struct Strides2D {
    ptrdiff_t head, seq;

    __device__ ptrdiff_t offset(size_t head_, size_t seq_) const {
        return head_ * head + seq_ * seq;
    }
};

struct KernelReq {
    Tdata const *q;
    Strides2D q_strides;
    size_t pages_start;
    Strides2D kv_strides;
    Tdata *o;
    Strides2D o_strides;
    bool *const mask;
    Tdata *l, *m;
    size_t n, s;
};

extern "C" __global__ void flash_attn(
    KernelCfg cfg,
    KVPage const *cache_pages,
    KernelReq const *reqs) {
    size_t const
        ireq = blockIdx.y,
        head = blockIdx.x,
        bn = blockDim.x,
        it = threadIdx.x;

    size_t const
        g = cfg.g,
        d = cfg.d,
        bs = cfg.bs;
    Tdata const
        scale = cfg.scale;

    KernelReq const
        req = reqs[ireq];
    KVPage const *
        pages = cache_pages + req.pages_start;

    extern __shared__ Tdata shared[];
    Tdata *qi = shared,
          *kj = qi + bn * d,
          *vj = kj + bs * d,
          *x = vj + bs * d;

    size_t const ikvb_end = div_ceil(req.s, bs);
    for (size_t ikvb = 0; ikvb < ikvb_end; ++ikvb) {
        Tdata const
            *k = (cache_pages + req.pages_start + ikvb)->k,
            *v = (cache_pages + req.pages_start + ikvb)->v;

        for (size_t i = it; i < bs; i += bn) {
            ptrdiff_t const offset = req.kv_strides.offset(head / g, i);
            memcpy(kj + i * d, byte_offset(k, offset), d * sizeof(Tdata));
            memcpy(vj + i * d, byte_offset(v, offset), d * sizeof(Tdata));
        }
        __syncthreads();
        {
            // TODO
        }
        __syncthreads();
    }
}
