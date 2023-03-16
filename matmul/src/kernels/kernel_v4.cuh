#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>

template <typename scalar_t, uint32_t BM, uint32_t BK, uint32_t BN>
__global__ void kernel_v4(
    const scalar_t * __restrict__ A,
    const scalar_t * __restrict__ B,
    uint32_t M, uint32_t K, uint32_t N,
    scalar_t * C 
) {
	// in v4, we take advantage of shared memory for more efficient access compared to global memory.

    // block index
    const uint32_t bm = blockIdx.y;
    const uint32_t bn = blockIdx.x;

    // allocate shared memory 
    // pass in blockDim.x/y as template parameter, since blockDim.x is not constant even blockDim itself is...
    __shared__ scalar_t AA[BM * BK];
    __shared__ scalar_t BB[BK * BN];

    // inside-block index
    const uint32_t m = threadIdx.y;
    const uint32_t n = threadIdx.x;

    // advance pointers
    A += K * bm * BM;
    B += bn * BN;
    C += N * bm * BM + bn * BN + N * m + n;

    scalar_t res = 0;
    // iterate on block
    for (uint32_t i = 0; i < K; i += BK) {
        // write to shared memory
        AA[m * BK + n] = A[m * K + n];
        BB[m * BN + n] = B[m * N + n];
        __syncthreads(); // make sure all threads have written to shared memory

        // iterate inside block, and accumulate along K
        for (uint32_t j = 0; j < BK; j++) {
            res += AA[m * BK + j] * BB[j * BN + n];
        }
        __syncthreads(); // make sure all threads have finished using shared memory

        // advance pointers for loading the next block into shared memory
        A += BK;
        B += BK * N;
    }
    C[0] = res;
}
