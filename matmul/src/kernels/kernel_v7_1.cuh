#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>

template <typename scalar_t, uint32_t BM, uint32_t BK, uint32_t BN, uint32_t T>
__global__ void kernel_v7_1(
    scalar_t * __restrict__ A, // note that we remove const to make A castable to float4
    scalar_t * __restrict__ B,
    uint32_t M, uint32_t K, uint32_t N,
    scalar_t * C 
) {
    // in v7, we apply duble buffering.

    // block index
    const uint32_t bm = blockIdx.y;
    const uint32_t bn = blockIdx.x;

    // allocate shared memory 
    // BK < BM or BN, as the total size of shared memory is limited.
    __shared__ scalar_t AA[2][BM * BK];
    __shared__ scalar_t BB[2][BK * BN];

    // inside-block index
    const uint32_t m = threadIdx.y;
    const uint32_t n = threadIdx.x;

    // advance pointers
    A += K * bm * BM;
    B += bn * BN;
    C += N * (bm * BM + m * T) + bn * BN + n * T;

    scalar_t res[T * T] = {0};

    // allocate registers
    scalar_t rA[2][T], rB[2][T];

    // as all threads are supposed to process BM * BN entries of C per block, 
    // but AA (BB) only need to load BM * BK (BN * BK) < BM * BN entries, 
    // we would like to rearrange the threads to balance the workload.
    const uint32_t num_threads = (BM / T) * (BN / T); // 64
    const uint32_t b = m * (BM / T) + n; // assume BM * BK (BN * BK) is divisible by num_threads
    const uint32_t mA = b / (BK / 4), mB = b / (BN / 4);
    const uint32_t nA = b % (BK / 4), nB = b % (BN / 4);
    const uint32_t sA = num_threads / (BK / 4), sB = num_threads / (BN / 4);

    const uint32_t step_A = BM / sA, step_B = BK / sB;
    float4 rsA[step_A], rsB[step_B];

    // preload
    for (uint32_t t = 0; t < BM; t += sA) {
        float4 r = reinterpret_cast<float4 *>(&A[(mA + t) * K + nA * 4])[0];
        // transpose AA
        AA[0][(nA * 4 + 0) * BM + mA + t] = r.x;
        AA[0][(nA * 4 + 1) * BM + mA + t] = r.y;
        AA[0][(nA * 4 + 2) * BM + mA + t] = r.z;
        AA[0][(nA * 4 + 3) * BM + mA + t] = r.w;
    }

    for (uint32_t t = 0; t < BK; t += sB) {
        reinterpret_cast<float4 *>(&BB[0][(mB + t) * BN + nB * 4])[0] = \
        reinterpret_cast<float4 *>(&B[(mB + t) * N + nB * 4])[0];
    }

    __syncthreads();


    for (uint32_t t = 0; t < T; t += 4) {
        reinterpret_cast<float4 *>(&rA[0][t])[0] = reinterpret_cast<float4 *>(&AA[0][m * T + t])[0];
        reinterpret_cast<float4 *>(&rB[0][t])[0] = reinterpret_cast<float4 *>(&BB[0][n * T + t])[0];
    }

    A += BK;
    B += BK * N;

    // load/write index
    uint32_t write_id = 1, load_id = 0;

    // iterate on block
    for (uint32_t i = 0; i < K - BK; i += BK) {

        // load rsA, rsB
        for (uint32_t t = 0; t < step_A; t++) {
            rsA[t] = reinterpret_cast<float4 *>(&A[(mA + t * sA) * K + nA * 4])[0];
        }
        for (uint32_t t = 0; t < step_B; t++) {
            rsB[t] = reinterpret_cast<float4 *>(&B[(mB + t * sB) * N + nB * 4])[0];
        }

        // iterate inside block, and accumulate along K
        // note we only loop to j = BK - 2, as the last iteration should not write to register and handled specially.
        for (uint32_t j = 0; j < BK - 1; j++) {
            
            // write to register
            for (uint32_t t = 0; t < T; t += 4) {
                reinterpret_cast<float4 *>(&rA[(j + 1) % 2][t])[0] = reinterpret_cast<float4 *>(&AA[load_id][m * T + (j + 1) * BM + t])[0];
                reinterpret_cast<float4 *>(&rB[(j + 1) % 2][t])[0] = reinterpret_cast<float4 *>(&BB[load_id][n * T + (j + 1) * BN + t])[0];
            }

            for (uint32_t ti = 0; ti < T; ti++) {
                for (uint32_t tj = 0; tj < T; tj++) {
                    res[ti * T + tj] += rA[j % 2][ti] * rB[j % 2][tj];
                }
            }
        }

        // __syncthreads(); // thanks to double buffering, we do not need to sync here!

        // write to shared memory
        for (uint32_t t = 0; t < step_A; t++) {
            // transpose AA
            AA[write_id][(nA * 4 + 0) * BM + mA + t * sA] = rsA[t].x;
            AA[write_id][(nA * 4 + 1) * BM + mA + t * sA] = rsA[t].y;
            AA[write_id][(nA * 4 + 2) * BM + mA + t * sA] = rsA[t].z;
            AA[write_id][(nA * 4 + 3) * BM + mA + t * sA] = rsA[t].w;
        }

        for (uint32_t t = 0; t < step_B; t++) {
            reinterpret_cast<float4 *>(&BB[write_id][(mB + t * sB) * BN + nB * 4])[0] = rsB[t];
        }
        
        __syncthreads(); // make sure all threads have written to shared memory

        for (uint32_t t = 0; t < T; t += 4) {
            reinterpret_cast<float4 *>(&rA[0][t])[0] = reinterpret_cast<float4 *>(&AA[write_id][m * T + t])[0];
            reinterpret_cast<float4 *>(&rB[0][t])[0] = reinterpret_cast<float4 *>(&BB[write_id][n * T + t])[0];
        }

        // the last loop (j = BK - 1), assert BK % 2 == 0...
        for (uint32_t ti = 0; ti < T; ti++) {
            for (uint32_t tj = 0; tj < T; tj++) {
                res[ti * T + tj] += rA[1][ti] * rB[1][tj];
            }
        }

        // advance pointers for loading the next block into shared memory
        A += BK;
        B += BK * N;
        write_id ^= 1;
        load_id ^= 1;
    }

    // the last loop (i = K - BK)
    for (uint32_t j = 0; j < BK - 1; j++) {
        // write to register
        for (uint32_t t = 0; t < T; t++) {
            rA[(j + 1) % 2][t] = AA[load_id][m * T + (j + 1) * BM + t];
            rB[(j + 1) % 2][t] = BB[load_id][n * T + (j + 1) * BN + t];
        }

        for (uint32_t ti = 0; ti < T; ti++) {
            for (uint32_t tj = 0; tj < T; tj++) {
                res[ti * T + tj] += rA[j % 2][ti] * rB[j % 2][tj];
            }
        }
    }

    for (uint32_t ti = 0; ti < T; ti++) {
        for (uint32_t tj = 0; tj < T; tj++) {
            res[ti * T + tj] += rA[1][ti] * rB[1][tj];
        }
    }

    for (uint32_t ti = 0; ti < T; ti++) {
        for (uint32_t tj = 0; tj < T; tj += 4) {
            reinterpret_cast<float4 *>(&C[ti * N + tj])[0] = \
            reinterpret_cast<float4 *>(&res[ti * T + tj])[0];
        }
    }
}
