#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>

template <typename scalar_t, uint32_t BM, uint32_t BK, uint32_t BN, uint32_t T>
__global__ void kernel_v6(
    scalar_t * __restrict__ A, // note that we remove const to make A castable to float4
    scalar_t * __restrict__ B,
    uint32_t M, uint32_t K, uint32_t N,
    scalar_t * C 
) {
    // in v6, we use float4 to vectorize memory access.

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
    const uint32_t num_threads = (BM / T) * (BN / T); // 64
    const uint32_t b = m * (BM / T) + n; // assume BM * BK (BN * BK) is divisible by num_threads
    const uint32_t mA = b / (BK / 4), mB = b / (BN / 4);
    const uint32_t nA = b % (BK / 4), nB = b % (BN / 4);
    const uint32_t sA = num_threads / (BK / 4), sB = num_threads / (BN / 4);

    // iterate on block
    for (uint32_t i = 0; i < K; i += BK) {

        // write to shared memory
        #pragma unroll
        for (uint32_t t = 0; t < BM; t += sA) {
            float4 r = reinterpret_cast<float4 *>(&A[(mA + t) * K + nA * 4])[0];
            // transpose AA
            AA[(nA * 4 + 0) * BM + mA + t] = r.x;
            AA[(nA * 4 + 1) * BM + mA + t] = r.y;
            AA[(nA * 4 + 2) * BM + mA + t] = r.z;
            AA[(nA * 4 + 3) * BM + mA + t] = r.w;
        }

        #pragma unroll
        for (uint32_t t = 0; t < BK; t += sB) {
            reinterpret_cast<float4 *>(&BB[(mB + t) * BN + nB * 4])[0] = \
            reinterpret_cast<float4 *>(&B[(mB + t) * N + nB * 4])[0];
        }
        
        __syncthreads(); // make sure all threads have written to shared memory

        // iterate inside block, and accumulate along K
        #pragma unroll
        for (uint32_t j = 0; j < BK; j++) {
            
            // write to register
            #pragma unroll
            for (uint32_t t = 0; t < T; t += 4) {
                reinterpret_cast<float4 *>(&rA[t])[0] = reinterpret_cast<float4 *>(&AA[m * T + j * BM + t])[0];
                reinterpret_cast<float4 *>(&rB[t])[0] = reinterpret_cast<float4 *>(&BB[n * T + j * BN + t])[0];
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
        for (uint32_t tj = 0; tj < T; tj += 4) {
            reinterpret_cast<float4 *>(&C[ti * N + tj])[0] = \
            reinterpret_cast<float4 *>(&res[ti * T + tj])[0];
        }
    }
}
