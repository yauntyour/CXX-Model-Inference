"""Prototype: give the official AstrAI model our W8A16 GEMV for M=1 decode.

Builds a small CUDA extension (our warp-per-row kernel), patches F.linear to
use it when the flattened token count is 1, then runs the official
GenerationBenchmark. Prefill (M>1) stays on the official BF16 path.
"""

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repos" / "AstrAI"
MODEL = Path(r"D:\models\AstrAI-V1-2B-instruct")
sys.path.insert(0, str(REPO))

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }

template <int WARPS>
__global__ void gemv_w8a16_kernel(const bf16* __restrict__ x,
                                  const int8_t* __restrict__ Q,
                                  const float* __restrict__ scale,
                                  bf16* __restrict__ y, int N, int K,
                                  const bf16* __restrict__ add) {
    const int warp = blockIdx.x * WARPS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (warp >= N) return;
    const int8_t* row = Q + (size_t)warp * K;
    float acc = 0.f;
    const bool aligned = (K % 16 == 0) &&
        ((((uintptr_t)row | (uintptr_t)x) & 15) == 0);
    if (aligned) {
        int i = lane * 16;
        for (; i + 15 < K; i += 512) {
            uint4 wv = *reinterpret_cast<const uint4*>(row + i);
            uint4 xv0 = *reinterpret_cast<const uint4*>(x + i);
            uint4 xv1 = *reinterpret_cast<const uint4*>(x + i + 8);
            const int8_t* wb = reinterpret_cast<const int8_t*>(&wv);
            const __nv_bfloat162* xb0 = reinterpret_cast<const __nv_bfloat162*>(&xv0);
            const __nv_bfloat162* xb1 = reinterpret_cast<const __nv_bfloat162*>(&xv1);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 xf = __bfloat1622float2(xb0[k]);
                acc += (float)wb[2*k] * xf.x + (float)wb[2*k+1] * xf.y;
            }
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 xf = __bfloat1622float2(xb1[k]);
                acc += (float)wb[8+2*k] * xf.x + (float)wb[9+2*k] * xf.y;
            }
        }
        for (int j = i; j < K; j++) acc += (float)row[j] * b2f(x[j]);
    } else {
        for (int j = lane; j < K; j += 32) acc += (float)row[j] * b2f(x[j]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) {
        float v = acc * scale[warp];
        if (add) v += b2f(add[warp]);
        y[warp] = f2b(v);
    }
}

torch::Tensor w8a16_gemv(torch::Tensor x, torch::Tensor q, torch::Tensor scale,
                         c10::optional<torch::Tensor> add) {
    TORCH_CHECK(x.is_cuda() && q.is_cuda() && scale.is_cuda(), "cuda tensors required");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(q.scalar_type() == torch::kChar, "q must be int8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be fp32");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "M=1 only");
    const int K = (int)x.size(1);
    const int N = (int)q.size(0);
    TORCH_CHECK((int)q.size(1) == K, "q shape mismatch");
    TORCH_CHECK((int)scale.numel() == N, "scale shape mismatch");
    auto qc = q.contiguous();
    auto sc = scale.contiguous();
    auto xc = x.contiguous();
    auto y = torch::empty({1, N}, x.options());
    const bf16* addp = nullptr;
    c10::optional<torch::Tensor> addc;
    if (add.has_value()) {
        addc = add->to(torch::kBFloat16).contiguous();
        TORCH_CHECK((int)addc->numel() == N, "add shape mismatch");
        addp = reinterpret_cast<const bf16*>(addc->data_ptr());
    }
    constexpr int WARPS = 8;
    int blocks = (N + WARPS - 1) / WARPS;
    gemv_w8a16_kernel<WARPS><<<blocks, WARPS * 32, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const bf16*>(xc.data_ptr()),
        qc.data_ptr<int8_t>(), sc.data_ptr<float>(),
        reinterpret_cast<bf16*>(y.data_ptr()), N, K, addp);
    return y;
}
"""

CPP_DECL = (
    "#include <torch/extension.h>\n"
    "torch::Tensor w8a16_gemv(torch::Tensor x, torch::Tensor q,"
    " torch::Tensor scale, c10::optional<torch::Tensor> add);\n"
)

ext = load_inline(
    name="astrai_w8a16_gemv",
    cpp_sources=CPP_DECL,
    cuda_sources=CUDA_SRC,
    functions=["w8a16_gemv"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xcompiler=/Zc:preprocessor"],
    verbose=False,
)

_qcache = {}
_orig_linear = F.linear
_stat_calls = 0
_stat_miss = 0
_stat_fallback = 0
_stat_rows = {}


def _qlinear(x, weight, bias=None):
    global _stat_calls, _stat_miss, _stat_fallback
    rows = x.numel() // x.shape[-1] if x.numel() else 0
    _stat_rows[rows] = _stat_rows.get(rows, 0) + 1
    if (
        not x.is_cuda
        or weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or x.shape[-1] != weight.shape[1]
    ):
        _stat_fallback += 1
        return _orig_linear(x, weight, bias)
    _stat_calls += 1
    if rows != 1:
        _stat_fallback += 1
        return _orig_linear(x, weight, bias)
    key = (weight.data_ptr(), tuple(weight.shape), weight.device)
    if key not in _qcache:
        _stat_miss += 1
        w = weight.float()
        s = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        q = (w / s[:, None]).round().clamp(-127, 127).to(torch.int8).contiguous()
        _qcache[key] = (q, s.contiguous())
    q, s = _qcache[key]
    x2 = x.reshape(1, -1).contiguous()
    y = ext.w8a16_gemv(x2, q, s, bias)
    return y.reshape(*x.shape[:-1], weight.shape[0])


import os as _os
if _os.environ.get("ASTRAI_NO_PATCH") != "1":
    F.linear = _qlinear

from astrai.config import ConfigFactory  # noqa: E402
from astrai.extension.runtime.context import op_backend  # noqa: E402
from astrai.inference.core.cache import BlockPool  # noqa: E402
from astrai.model import AutoModel  # noqa: E402
from astrai.serialization import adapt_config  # noqa: E402
from astrai.tokenize import AutoTokenizer  # noqa: E402
from astrai.inference.frontend.engine import InferenceEngine  # noqa: E402
from astrai.inference.worker.graph import CUDAGraphRunner  # noqa: E402
from astrai.inference.worker.workspace import InferenceWorkspace  # noqa: E402
from astrai.inference.core.cache import KVCacheManager  # noqa: E402


def main():
    raw = json.loads((MODEL / "config.json").read_text(encoding="utf-8-sig"))
    config = ConfigFactory.load(adapt_config(raw, str(MODEL)))
    model = AutoModel.from_pretrained(str(MODEL))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    model.to("cuda", dtype=torch.bfloat16)
    model.eval()

    # Warm up the W8A16 caches before graph capture.
    for m in model.modules():
        w = getattr(m, "weight", None)
        if isinstance(w, torch.nn.Parameter) and w.dim() == 2:
            xw = torch.zeros(1, w.shape[1], device="cuda", dtype=torch.bfloat16)
            _qlinear(xw, w, getattr(m, "bias", None))
    torch.cuda.synchronize()

    from scripts.benchmark.inference import (  # noqa: E402
        GenerationBenchmark,
        print_benchmark_result,
    )

    bench = GenerationBenchmark(
        model=model,
        config=config,
        device="cuda",
        dtype=torch.bfloat16,
        cache_type="contiguous",
        backend="cuda",
        cuda_graph=os.environ.get("ASTRAI_NO_GRAPH") != "1",
        tokenizer=tokenizer,
    )
    short = os.environ.get("ASTRAI_W8_SHORT") == "1"
    prompt_lens = (128,) if short else (128, 512)
    gen_len = 8 if short else 32
    trials = 1 if short else 3
    for prompt_len in prompt_lens:
        r = bench.run_decoding_benchmark(
            batch_size=1, prompt_length=prompt_len, gen_length=gen_len,
            num_trials=trials, overlap=False,
        )
        print_benchmark_result(r)
    if not short:
        for kv_len in (128, 512):
            r = bench.run_fixed_kv_decode_benchmark(
                batch_size=1, kv_len=kv_len, window_steps=32
            )
            print_benchmark_result(r)

    print(
        "[w8a16 stats] calls",
        _stat_calls,
        "cache misses",
        _stat_miss,
        "fallbacks",
        _stat_fallback,
        "rows",
        _stat_rows,
        "cache",
        len(_qcache),
    )


if __name__ == "__main__":
    main()
