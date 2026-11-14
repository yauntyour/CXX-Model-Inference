import torch
import torch.nn.functional as F
import time
from astrai.extension.kernel.gemm import quant_gemm

torch.manual_seed(0)
shapes = [
    (1, 3072, 2048),
    (1, 2048, 2048),
    (1, 16384, 2048),
    (1, 2048, 8192),
    (1, 131072, 2048),
    (128, 3072, 2048),
    (128, 16384, 2048),
    (512, 16384, 2048),
]

def bench(fn, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

for M, N, K in shapes:
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    q = torch.randint(-127, 127, (K, N), device="cuda", dtype=torch.int8)
    s = (torch.rand(N, device="cuda") * 0.01 + 1e-4)
    tq = bench(lambda: quant_gemm(a, q, None, s, trans_b=False))
    tb = bench(lambda: F.linear(a, w))
    print(f"M={M} N={N} K={K}: w8a16 {tq:.3f} ms, bf16 {tb:.3f} ms, ratio {tb/tq:.2f}x")
