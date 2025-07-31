template <size_t D, size_t WARP, typename T>
__device__ __forceinline__ void load_qo_(
    T *qi,
    T *oi,
    T const *q,
    T const zero) {
#pragma unroll
    for (size_t i = threadIdx.x; i < D; i += WARP) {
        qi[i] = q[i]; // 加载 qi
        oi[i] = zero; // 初始化 oi 为 0
    }
}

template <size_t D, size_t WARP, typename T>
__device__ __forceinline__ void load_kv_(
    T *kj,
    T *vj,
    T const *k,
    T const *v) {
#pragma unroll
    for (size_t i = threadIdx.x; i < D; i += WARP) {
        kj[i] = k[i];
        vj[i] = v[i];
    }
}
