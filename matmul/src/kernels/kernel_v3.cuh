#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>

template <typename scalar_t>
__global__ void kernel_v3(
    const scalar_t * __restrict__ A,
    const scalar_t * __restrict__ B,
    uint32_t M, uint32_t K, uint32_t N,
    scalar_t * C 
) {
    // the only change from v2 is the order of thread loops, which leads to 10x acceleration.
    // the reason is that the global memory access pattern is more coalesced.
    // by coalescing, we let consecutive threads access consecutive memory locations.
    // i.e., for consecutive threads threadIdx.x = 0, 1, 2, ... in a warp (every 32 threads),
    //    threadIdx.x == 0: access A[m, :] and B[:, n]
    //    threadIdx.x == 1: access A[m, :] and B[:, n + 1]
    //    ...
    // B[k, n], B[k, n + 1], ... are consecutive memory locations (as B is row-major),
    // so the warp can group them and execute memory access as one! and this is where we gain speed.
    const uint32_t m = threadIdx.y + blockIdx.x * blockDim.x;
    const uint32_t n = threadIdx.x + blockIdx.y * blockDim.y;
    if (m >= M || n >= N) return;

    A += K * m;
    B += n;
    C += N * m + n;

    scalar_t res = 0;
    for (uint32_t i = 0; i < K; i++) {
        res += A[i] * B[i * N];
    }
    C[0] = res;
}
