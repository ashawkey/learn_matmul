#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>

template <typename scalar_t, uint32_t BM, uint32_t BK, uint32_t BN, uint32_t T>
__global__ void kernel_v5_2(
    const scalar_t * __restrict__ A,
    const scalar_t * __restrict__ B,
    uint32_t M, uint32_t K, uint32_t N,
    scalar_t * C 
) {
    // in v5, we calculate multiple elements of C in one thread, as v4 is bound by shared memory access.

    // block index
    const uint32_t bm = blockIdx.y;
    const uint32_t bn = blockIdx.x;

    // allocate shared memory 
    // BK < BM or BN, as the total size of shared memory is limited.
    __shared__ scalar_t AA[BM * BK];
    __shared__ scalar_t BB[BK * BN];

    // inside-block index
    const uint32_t m = threadIdx.y;
    const uint32_t n = threadIdx.x;

    // advance pointers
    A += K * bm * BM;
    B += bn * BN;
    C += N * (bm * BM + m * T) + bn * BN + n * T;

    scalar_t res[T * T] = {0};

    // allocate registers
    scalar_t rA[T], rB[T];

    // as all threads are supposed to process BM * BN entries of C per block, 
    // but AA (BB) only need to load BM * BK (BN * BK) < BM * BN entries, 
    // we would like to rearrange the threads to balance the workload.
    const uint32_t num_threads = blockDim.x * blockDim.y; // 64
    const uint32_t b = m * blockDim.x + n; // assume BM * BK (BN * BK) is divisible by num_threads
    const uint32_t mA = b / BK, mB = b / BN;
    const uint32_t nA = b % BK, nB = b % BN;
    const uint32_t sA = num_threads / BK, sB = num_threads / BN;

    // iterate on block
    for (uint32_t i = 0; i < K; i += BK) {

        // write to shared memory
        #pragma unroll
        for (uint32_t t = 0; t < BM; t += sA) {
            // transpose AA
            AA[nA * BM + mA + t] = A[(mA + t) * K + nA];
        }
        #pragma unroll
        for (uint32_t t = 0; t < BK; t += sB) {
            BB[(mB + t) * BN + nB] = B[(mB + t) * N + nB];
        }
        
        __syncthreads(); // make sure all threads have written to shared memory

        // iterate inside block, and accumulate along K
        #pragma unroll
        for (uint32_t j = 0; j < BK; j++) {
            
            // write to register
            #pragma unroll
            for (uint32_t t = 0; t < T; t++) {
                rA[t] = AA[m * T + j * BM + t];
                rB[t] = BB[n * T + j * BN + t];
            }

            #pragma unroll
            for (uint32_t ti = 0; ti < T; ti++) {
                #pragma unroll
                for (uint32_t tj = 0; tj < T; tj++) {
                    res[ti * T + tj] += rA[ti] * rB[tj];
                }
            }
        }
        __syncthreads(); // make sure all threads have finished using shared memory

        // advance pointers for loading the next block into shared memory
        A += BK;
        B += BK * N;
    }

    #pragma unroll
    for (uint32_t ti = 0; ti < T; ti++) {
        #pragma unroll
        for (uint32_t tj = 0; tj < T; tj++) {
            C[ti * N + tj] = res[ti * T + tj];
        }
    }
}
