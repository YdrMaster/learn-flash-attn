template <size_t D, typename T>
__device__ __forceinline__ void load_qo_(
    T *qi,
    T *oi,
    T const *q,
    T const zero) {
    size_t const
        warp = blockDim.x,
        lane = threadIdx.x;
    for (size_t i = lane; i < D; i += warp) {
        qi[i] = q[i]; // 加载 qi
        oi[i] = zero; // 初始化 oi 为 0
    }
}

template <size_t D, typename T>
__device__ __forceinline__ void load_kv_(
    T *kj,
    T *vj,
    T const *k,
    T const *v) {
    size_t const
        warp = blockDim.x,
        lane = threadIdx.x;
    for (size_t i = lane; i < D; i += warp) {
        kj[i] = k[i];
        vj[i] = v[i];
    }
}
