#pragma once
// ============================================================
// 推理快路径算子 (参考 MetaModel 的 fast 层设计)
//
// 与 kernels.hpp 的分工:
//   kernels.* = 朴素参考实现 (逐 token / 逐 head 循环)
//   fast_ops  = 面向 decode 的 GEMV、面向 prefill 的 NT GEMM、
//               融合 QKV+RoPE+KV 写入、融合 SwiGLU、warp-per-head 注意力
//
// 权重统一保持 PyTorch 原始布局 [out, in]，不再做 CPU 端转置:
//   - prefill: cuBLAS GemmEx(opA=T) 直接吃 [out, in]
//   - decode : 自写 GEMV 每个 warp 读一整行, 合并访存
// ============================================================

#include "kernels.hpp"

namespace fo
{
    // y[n] = dot(x, W[n, :]) (+ add[n] if add != nullptr)
    // W: [N, K] 行主序, x: [K], y/add: [N]
    void gemv_bf16(const qk::bf16* x, const qk::bf16* W, qk::bf16* y,
                   int N, int K, const qk::bf16* add = nullptr,
                   cudaStream_t s = 0);

    // ------------------------------------------------------------
    // W8A16: 权重 int8 (per-row symmetric), 激活保持 bf16
    // ------------------------------------------------------------
    // W[N, K] bf16 -> Q[N, K] int8 + scale[N] fp32
    void quantize_w8a16(const qk::bf16* W, int8_t* Q, float* scale,
                        int N, int K, cudaStream_t s = 0);
    // y[n] = (sum_k Q[n,k] * x[k]) * scale[n] (+ add[n])
    void gemv_w8a16(const qk::bf16* x, const int8_t* Q, const float* scale,
                    qk::bf16* y, int N, int K, const qk::bf16* add = nullptr,
                    cudaStream_t s = 0);

    // C[M, N] = A[M, K] * W[N, K]^T + beta * C
    // M>1 用 cuBLAS; M==1 自动走 GEMV
    void gemm_nt_bf16(const qk::bf16* A, const qk::bf16* W, qk::bf16* C,
                      int M, int N, int K, float beta = 0.0f,
                      cudaStream_t s = 0);

    // 融合: qkv 投影结果按头拆分 + Q/K RoPE + K/V 直接写入 KV cache
    // qkv: [S, (Hq+2*Hkv)*D], 每 token 内 [Q | K | V]
    // pos: [S] 绝对位置; inv_freq: [rot/2]
    // q_out: [S, Hq*D]; kc/vc: [max_ctx, Hkv*D]
    // interleaved: 0=Llama 半维配对, 1=GPT-J 交错配对
    void qkv_rope_kv(const qk::bf16* qkv, const int* pos, const float* inv_freq,
                     qk::bf16* q_out, qk::bf16* kc, qk::bf16* vc,
                     int base, int S, int Hq, int Hkv, int D, int rot,
                     int interleaved, cudaStream_t s = 0);

    // out[r, i] = silu(gu[r, i]) * gu[r, I + i]
    void swiglu_bf16(const qk::bf16* gu, qk::bf16* out, int rows, int I,
                     cudaStream_t s = 0);

    // decode 注意力: q [1, Hq*D], kc/vc [len, Hkv*D], out [1, Hq*D]
    // 一个 block 对应一个 KV head; block 内每个 warp 负责该组的一个 Q head.
    void decode_attn(const qk::bf16* q, const qk::bf16* kc, const qk::bf16* vc,
                     qk::bf16* out, int len, int Hq, int Hkv, int D,
                     float scale, cudaStream_t s = 0);

    // prefill 闪存注意力 (bf16 WMMA, 两遍, D=128)
    // q: [S, Hq*D], kc/vc: [len, Hkv*D], out: [S, Hq*D]
    void flash_attn_prefill(const qk::bf16* q, const qk::bf16* kc, const qk::bf16* vc,
                            qk::bf16* out, int base, int S, int len, int Hq, int Hkv,
                            int D, float scale, cudaStream_t s = 0);

    // 判断 decode / prefill 注意力是否支持该 head_dim
    bool decode_attn_supported(int D);
    bool flash_attn_supported(int D);
}
