# pragma once

#include <stdint.h>
#include <torch/torch.h>

void matmul_forward(at::Tensor A, at::Tensor B, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t ver, at::Tensor C);
// void matmul_backward(at::Tensor grad_C, at::Tensor A, at::Tensor B, const uint32_t M, const uint32_t K, const uint32_t N, at::Tensor grad_A, at::Tensor grad_B);