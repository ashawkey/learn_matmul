import torch
import matmul
import argparse
import json
import os

# test kernel at different input size, and draw a plot.
def test_kernel(ver, atol=1e-2, repeat=10, dtype=torch.float32, scale=1):
    
    sizes = [256, 512, 1024, 2048, 4096]

    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)

    res = {}
    
    for size in sizes:
        
        # allocate
        A = scale * torch.randn(size, size, dtype=dtype, device='cuda')
        B = scale * torch.randn(size, size, dtype=dtype, device='cuda')
        C = torch.zeros(size, size, dtype=dtype, device='cuda')
        C_ref = torch.zeros(size, size, dtype=dtype, device='cuda')
        
        # test correctness & warm up
        torch.matmul(A, B, out=C_ref) # as gt

        if ver == 0:
            torch.matmul(A, B, out=C)
        else:
            matmul.matmul(A, B, C=C, ver=ver)

        torch.cuda.synchronize()
        if not torch.allclose(C_ref, C, atol=atol):
            error_mask = (C_ref - C).abs() > atol
            error_ind = torch.nonzero(error_mask)
            print('error entries: ', error_ind.shape[0], ' / ', size ** 2)
            print(C_ref)
            print(C)
            print('error indices and values:')
            print(error_ind[:10])
            print(C_ref[error_ind[:10, 0], error_ind[:10, 1]])
            print(C[error_ind[:10, 0], error_ind[:10, 1]])
            raise ValueError(f'incorrect result for ver={ver} size={size}!')


        # test speed
        tic.record()
        for _ in range(repeat):
            if ver == 0:
                torch.matmul(A, B, out=C)
            else:
                matmul.matmul(A, B, C=C, ver=ver)
        toc.record()

        torch.cuda.synchronize()
        total_t = tic.elapsed_time(toc) / 1000 # sec

        res[size] = {
            't': total_t / repeat,
            'gflops': 2 * 1e-9 * repeat * size ** 3 / total_t,
        }

        print(f'[INFO] ver = {ver} size = {size: 5} t = {res[size]["t"]:.6f} GFLPOs = {res[size]["gflops"]:.2f}')

    # save output
    os.makedirs('logs', exist_ok=True)
    with open(f"./logs/{ver}.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ver', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    opt = parser.parse_args()

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    test_kernel(opt.ver)
