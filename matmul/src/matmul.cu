#include <stdint.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cstdio>
#include <cassert>
#include <stdexcept>

#include "kernels/kernel_v1.cuh"
#include "kernels/kernel_v2.cuh"
#include "kernels/kernel_v3.cuh"
#include "kernels/kernel_v4.cuh"
#include "kernels/kernel_v5.cuh"
#include "kernels/kernel_v5_1.cuh"
#include "kernels/kernel_v5_2.cuh"
#include "kernels/kernel_v6.cuh"
#include "kernels/kernel_v7.cuh"
#include "kernels/kernel_v7_1.cuh"
#include "kernels/kernel_v7_2.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

#define DIV_ROUND_UP(X, Y) ((X) + (Y) - 1) / (Y)


void matmul_forward(at::Tensor A, at::Tensor B, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t ver, at::Tensor C) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CUDA(C);
    
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);
    CHECK_CONTIGUOUS(C);

    CHECK_IS_FLOATING(A);
    CHECK_IS_FLOATING(B);
    CHECK_IS_FLOATING(C);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    A.scalar_type(), "matmul_forward", ([&] {

        switch (ver) {
            // 1D
            case 1: {
                static constexpr uint32_t blockDim = 1024;
                uint32_t gridDim = DIV_ROUND_UP(M * N, blockDim);
                kernel_v1<scalar_t><<<gridDim, blockDim>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), M, K, N, C.data_ptr<scalar_t>());
            }; break;
            // 1D + less blockDim
            case 11: {
                static constexpr uint32_t blockDim = 256;
                uint32_t gridDim = DIV_ROUND_UP(M * N, blockDim);
                kernel_v1<scalar_t><<<gridDim, blockDim>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), M, K, N, C.data_ptr<scalar_t>());
            }; break;
            // 2D
            case 2: {
                static constexpr dim3 blockDim(32, 32);
                dim3 gridDim(DIV_ROUND_UP(M, blockDim.x), DIV_ROUND_UP(N, blockDim.y));
                kernel_v2<scalar_t><<<gridDim, blockDim>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), M, K, N, C.data_ptr<scalar_t>());
            }; break;
            // 2D + less blockDim
            case 21: {
                static constexpr dim3 blockDim(16, 16);
                dim3 gridDim(DIV_ROUND_UP(M, blockDim.x), DIV_ROUND_UP(N, blockDim.y));
                kernel_v2<scalar_t><<<gridDim, blockDim>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), M, K, N, C.data_ptr<scalar_t>());
            }; break;
            // 2D + global memory coalescing
            case 3: {
                static constexpr dim3 blockDim(32, 32);
                dim3 gridDim(DIV_ROUND_UP(M, blockDim.x), DIV_ROUND_UP(N, blockDim.y));
                kernel_v3<scalar_t><<<gridDim, blockDim>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), M, K, N, C.data_ptr<scalar_t>());
            }; break;
            // 2D + block tiling (shared memory)
            case 4: {
                static constexpr uint32_t BM = 32, BK = 32, BN = 32;
                // from this kernel, we give up handling border cases...
                assert(M % BM == 0 && N % BN == 0 && K % BK == 0);
                static constexpr dim3 blockDim(BM, BN);
                dim3 gridDim(DIV_ROUND_UP(M, BM), DIV_ROUND_UP(N, BN));
                kernel_v4<scalar_t, BM, BK, BN><<<gridDim, blockDim>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), M, K, N, C.data_ptr<scalar_t>());
            }; break;
            // 2D + block tiling + thread tiling (calc multiple elements of C per thread)
            case 5: {
                // BM, BK, BN must be divisible by T
                // BK is smaller, as we have limited registers per block
                static constexpr uint32_t BM = 128, BK = 16, BN = 128, T = 8;
                assert(M % BM == 0 && N % BN == 0 && K % BK == 0);
                dim3 blockDim(DIV_ROUND_UP(BM, T), DIV_ROUND_UP(BN, T));
                dim3 gridDim(DIV_ROUND_UP(M, BM), DIV_ROUND_UP(N, BN));
                kernel_v5<scalar_t, BM, BK, BN, T><<<gridDim, blockDim>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), M, K, N, C.data_ptr<scalar_t>());
            }; break;
            // 2D + block tiling + thread tiling + register
            case 51: {
                // BM, BK, BN must be divisible by T
                // BK is smaller, as we have limited registers per block
                static constexpr uint32_t BM = 128, BK = 16, BN = 128, T = 8;
                assert(M % BM == 0 && N % BN == 0 && K % BK == 0);
                dim3 blockDim(DIV_ROUND_UP(BM, T), DIV_ROUND_UP(BN, T));
                dim3 gridDim(DIV_ROUND_UP(M, BM), DIV_ROUND_UP(N, BN));
                kernel_v5_1<scalar_t, BM, BK, BN, T><<<gridDim, blockDim>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), M, K, N, C.data_ptr<scalar_t>());
            }; break;
            // 2D + block tiling + thread tiling + register + transpose
            case 52: {
                // BM, BK, BN must be divisible by T
                // BK is smaller, as we have limited registers per block
                static constexpr uint32_t BM = 128, BK = 16, BN = 128, T = 8;
                assert(M % BM == 0 && N % BN == 0 && K % BK == 0);
                dim3 blockDim(DIV_ROUND_UP(BM, T), DIV_ROUND_UP(BN, T));
                dim3 gridDim(DIV_ROUND_UP(M, BM), DIV_ROUND_UP(N, BN));
                kernel_v5_2<scalar_t, BM, BK, BN, T><<<gridDim, blockDim>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), M, K, N, C.data_ptr<scalar_t>());
            }; break;
            // 2D + block tiling + thread tiling + vectorize (float4)
            case 6: {
                // BM, BK, BN must be divisible by T
                // BK is smaller, as we have limited registers per block
                static constexpr uint32_t BM = 128, BK = 16, BN = 128, T = 8;
                assert(M % BM == 0 && N % BN == 0 && K % BK == 0);
                dim3 blockDim(DIV_ROUND_UP(BM, T), DIV_ROUND_UP(BN, T));
                dim3 gridDim(DIV_ROUND_UP(M, BM), DIV_ROUND_UP(N, BN));
                kernel_v6<scalar_t, BM, BK, BN, T><<<gridDim, blockDim>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), M, K, N, C.data_ptr<scalar_t>());
            }; break;
            // 2D + block tiling + thread tiling + vectorize + double buffer (do while impl)
            case 7: {
                // BM, BK, BN must be divisible by T
                // BK is smaller, as we have limited registers per block
                static constexpr uint32_t BM = 128, BK = 8, BN = 128, T = 8;
                assert(M % BM == 0 && N % BN == 0 && K % BK == 0);
                dim3 blockDim(DIV_ROUND_UP(BM, T), DIV_ROUND_UP(BN, T));
                dim3 gridDim(DIV_ROUND_UP(M, BM), DIV_ROUND_UP(N, BN));
                kernel_v7<scalar_t, BM, BK, BN, T><<<gridDim, blockDim>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), M, K, N, C.data_ptr<scalar_t>());
            }; break;
            // 2D + block tiling + thread tiling + vectorize + double buffer (for impl, w/o unroll) 
            // we'll see w/o pragma unroll it's VERY slow...
            case 71: {
                // BM, BK, BN must be divisible by T
                // BK is smaller, as we have limited registers per block
                static constexpr uint32_t BM = 128, BK = 8, BN = 128, T = 8;
                assert(M % BM == 0 && N % BN == 0 && K % BK == 0);
                dim3 blockDim(DIV_ROUND_UP(BM, T), DIV_ROUND_UP(BN, T));
                dim3 gridDim(DIV_ROUND_UP(M, BM), DIV_ROUND_UP(N, BN));
                kernel_v7_1<scalar_t, BM, BK, BN, T><<<gridDim, blockDim>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), M, K, N, C.data_ptr<scalar_t>());
            }; break;
            // 2D + block tiling + thread tiling + vectorize + double buffer (for impl, w/ unroll)
            case 72: {
                // BM, BK, BN must be divisible by T
                // BK is smaller, as we have limited registers per block
                static constexpr uint32_t BM = 128, BK = 8, BN = 128, T = 8;
                assert(M % BM == 0 && N % BN == 0 && K % BK == 0);
                dim3 blockDim(DIV_ROUND_UP(BM, T), DIV_ROUND_UP(BN, T));
                dim3 gridDim(DIV_ROUND_UP(M, BM), DIV_ROUND_UP(N, BN));
                kernel_v7_2<scalar_t, BM, BK, BN, T><<<gridDim, blockDim>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), M, K, N, C.data_ptr<scalar_t>());
            }; break;
            default: throw std::runtime_error{"Matmul: kernel version not implemented."};
        }
        
    }));	
}
