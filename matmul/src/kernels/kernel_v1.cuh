#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>

template <typename scalar_t>
__global__ void kernel_v1(
    const scalar_t * __restrict__ A,
    const scalar_t * __restrict__ B,
    uint32_t M, uint32_t K, uint32_t N,
    scalar_t * C 
) {
    const uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
    if (b >= M * N) return;

    const uint32_t m = b / N;
    const uint32_t n = b % N;

    A += K * m;
    B += n;
    C += b;

    scalar_t res = 0;
    for (uint32_t i = 0; i < K; i++) {
        res += A[i] * B[i * N];
    }
    C[0] = res;
}