## Steps to analyze CUDA kernel limitations

### Know your GPU specs.
Search them from google (usually a [whitepaper](https://images.nvidia.cn/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)).
We are most interested in the following specs (taking V100 as an example):
* (Single-Precision) Performance, e.g., 14T FLOPS.
* BandWidth, e.g., 900GB/s.
* Max Threads per block, e.g., 2048.
* Max Registers per block, e.g., 65536.
* Max Registers per thread, e.g., 255.
* Max Shared Memory per block, e.g., 96KB.

### Fastest possible runtime
Next we need to decide whether the kernel is compute- or memory-bound.
This requires per-algorithm per-input-size analysis, 
e.g., for matrix multiplication of two 4096x4096 matrices:
* FLOPS: 4096 ** 3 * 2 + 4096 ** 2 = 137G
* minimum read: 4096 ** 2 * 3 * 4B = 192MB
* minimum write: 4096 ** 2 * 4B = 64MB
these give us the following limits:
* compute: 137G / 14T = 9.79ms
* memory: (192MB + 64MB) / 900GB/s = 0.278ms
which shows the best possible kernel will be compute-bound in this input scale.

### Occupancy (TODO)

### Directions for optimization
* memory access pattern

### how to use tensorcore ? CUTLASS ? tiny-cuda-nn's wmma ?

