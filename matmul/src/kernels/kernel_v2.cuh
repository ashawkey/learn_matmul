#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>

template <typename scalar_t>
__global__ void kernel_v2(
    const scalar_t * __restrict__ A,
    const scalar_t * __restrict__ B,
    uint32_t M, uint32_t K, uint32_t N,
    scalar_t * C 
) {
	
    const uint32_t m = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t n = threadIdx.y + blockIdx.y * blockDim.y;
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
