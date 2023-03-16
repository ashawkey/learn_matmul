import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

import _matmul as _backend

class matmul_func(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, A, B, C=None, ver=0):
        # A: [M, K]
        # B: [K, N]
        # C: [M, N], output buffer if provided
        # ver: kernel version
        
        assert A.shape[1] == B.shape[0]

        M, K = A.shape
        K, N = B.shape
        
        if C is None:
            C = torch.empty(M, N, dtype=A.dtype, device=A.device)

        _backend.matmul_forward(A, B, M, K, N, ver, C)

        # ctx.save_for_backward(A, B)
        # ctx.dims = [M, K, N, ver]

        return C
    
    # TODO: backward
    # @staticmethod
    # @custom_bwd
    # def backward(ctx, grad_C):
    #     # grad_C: [M, N]

    #     A, B = ctx.saved_tensors
    #     M, K, N = ctx.dims

    #     grad_A = torch.empty_like(A)
    #     grad_B = torch.empty_like(B)

    #     _backend.matmul_backward(grad_C, A, B, M, K, N, grad_A, grad_B)

    #     return grad_A, grad_B


def matmul(A, B, C=None, ver=0):
    return matmul_func.apply(A, B, C, ver)