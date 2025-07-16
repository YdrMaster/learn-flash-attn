#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <math_constants.h>

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
    size_t const tn = div_ceil(req.n, bn);
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
        // 加载每个block私有的qi，x
        Tdata *qi_b = qi + it * d;
        Tdata *x_b = x + it * bs;

        for (uint64_t iqb = 0; iqb < tn; ++iqb) {
            uint64_t iq = iqb * bn + it;
            if (iq >= req.n) {
                continue;
            }
            // 加载数据
            memcpy(qi_b, byte_offset(req.q, req.q_strides.offset(head, iq)), d * sizeof(Tdata));
            Tdata *oi = byte_offset(req.o, req.o_strides.offset(head, iq));
            Tdata mi_1 = req.m[head * req.s + iq];
            Tdata di_1 = req.l[head * req.s + iq];
            bool const *mask = req.mask + iq * req.s + ikvb * bs;
            Tdata mi = mi_1;
            for (uint64_t i = 0; i < bs; ++i) {
                if (!mask[i]) {
                    x_b[i] = -CUDART_INF_F;
                } else {
                    x_b[i] = 0;
                    for (uint64_t j = 0; j < d; ++j) {
                        x_b[i] += qi_b[j] * kj[i * d + j];
                    }
                    x_b[i] *= scale;
                    if (x_b[i] > mi) {
                        mi = x_b[i];
                    }
                }
            }
            Tdata sum = 0.0;

            for (uint64_t i = 0; i < bs; ++i) {
                if (mask[i]) {
                    x_b[i] = exp(x_b[i] - mi);
                    sum += x_b[i];
                } else {
                    x_b[i] = 0;
                }
            }
            Tdata exp = di_1 * ::exp(mi_1 - mi);
            Tdata di = exp + sum;
            Tdata rdi = 1.0 / di;
            // 更新m，l
            req.m[head * req.s + iq] = mi;
            req.l[head * req.s + iq] = di;

            Tdata factor = exp * rdi;
            for (uint64_t j = 0; j < d; ++j) {
                Tdata v_acc = 0.0;
                for (uint64_t i = 0; i < bs; ++i) {
                    if (mask[i]) {
                        v_acc += x_b[i] * vj[i * d + j] * rdi;
                    }
                }
                oi[j] = oi[j] * factor + v_acc;
            }
        }
    }
}
