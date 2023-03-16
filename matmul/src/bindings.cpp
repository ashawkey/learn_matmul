#include <torch/extension.h>

#include "matmul.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_forward", &matmul_forward, "matmul forward (CUDA)");
    // m.def("matmul_backward", &matmul_backward, "matmul backward (CUDA)");
}