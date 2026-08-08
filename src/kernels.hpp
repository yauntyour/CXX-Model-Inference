#pragma once
// ============================================================
// qwen3.5 自定义 CUDA 内核声明 (kernels.cu 实现)
// 约定: 所有激活/权重均为 bf16 (qk::bf16, 2字节, 与 __nv_bfloat16 位兼容),
//       内核内部以 fp32 计算, 输出写回 bf16。
// ============================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace qk
{
    // 与 __nv_bfloat16 位兼容的 2 字节类型 (宿主/设备均可)
    typedef __nv_bfloat16 bf16;

    // ---------- 逐元素 ----------
    void silu(const bf16* x, bf16* y, int n, cudaStream_t s = 0);
    void sigmoid(const bf16* x, bf16* y, int n, cudaStream_t s = 0);
    void gelu_tanh(const bf16* x, bf16* y, int n, cudaStream_t s = 0);
    // 精确 GELU (erf)
    void gelu(const bf16* x, bf16* y, int n, cudaStream_t s = 0);
    // y = a * sigmoid(g)
    void sigmoid_mul(const bf16* a, const bf16* g, bf16* y, int n, cudaStream_t s = 0);
    // x += y (原地)
    void add_inplace(bf16* x, const bf16* y, int n, cudaStream_t s = 0);
    // x *= y (原地)
    void mul_inplace(bf16* x, const bf16* y, int n, cudaStream_t s = 0);
    // y = x + bias (每行加偏置)
    void add_bias(bf16* x, const bf16* bias, bf16* y, int n, int d, cudaStream_t s = 0);

    // g = -exp(A_log) * softplus(a + dt_bias), 按头: a: [n, H] bf16,
    // A_log: [H] fp32, dt_bias: [H] bf16 -> g: [n, H] bf16
    void gated_delta_params(const bf16* a, const float* A_log, const bf16* dt_bias,
                            bf16* g, int n, int H, cudaStream_t s = 0);

    // ---------- 归一化 (x: [n, d] 行主序) ----------
    // y = rmsnorm(x) * (1 + w)
    void rms_norm_1pw(const bf16* x, const bf16* w, bf16* y, int n, int d, float eps, cudaStream_t s = 0);
    // y = rmsnorm(x) * w * silu(gate)
    void rms_norm_gated(const bf16* x, const bf16* w, const bf16* gate, bf16* y,
                        int n, int d, float eps, cudaStream_t s = 0);
    // y = layernorm(x) * w + b
    void layer_norm(const bf16* x, const bf16* w, const bf16* b, bf16* y,
                    int n, int d, float eps, cudaStream_t s = 0);

    // ---------- embedding gather ----------
    // table: [V, d], ids: [n] -> out: [n, d]
    void embed_gather(const bf16* table, const int* ids, bf16* out, int n, int d, cudaStream_t s = 0);

    // ---------- q/gate 拆分 ----------
    // x: [n, H*2*D] 每 token 内按头交错 [q0 g0 q1 g1 ...] -> q/gate: [n, H*D]
    void split_qgate(const bf16* x, bf16* q, bf16* gate, int n, int H, int D,
                     cudaStream_t s = 0);

    // ---------- RoPE ----------
    // 文本 mrope: q: [n, H*D], 仅旋转前 rot_dim 维, 3 通道 (T/H/W) 交错
    // inv_freq: [n_freq], section: [3] (T/H/W 长度), pos_t/h/w: [n]
    void rope_text(const bf16* x, bf16* y, const int* pos_t, const int* pos_h, const int* pos_w,
                   const float* inv_freq, const int* section, int n, int H, int D, int rot_dim,
                   cudaStream_t s = 0);
    // 视觉 rope: x: [n, x_stride] 行主序交错 (每 token [q|k|v], 每段 H*D), 全维旋转;
    // y: [n, y_stride] 输出布局 (通常 H*D); pos: [n, 2] (h, w), inv_freq: [n_freq]
    void rope_vision(const bf16* x, bf16* y, const int* pos, const float* inv_freq,
                     int n, int H, int D, int x_stride, int y_stride, cudaStream_t s = 0);

    // ---------- 因果 depthwise conv1d (GatedDeltaNet) ----------
    // x: [seq, C], w: [C, K], state: [C, K-1] (in/out), y: [seq, C]
    // out[t] = silu( sum_k w[c,k] * x[t-K+1+k] )
    void conv1d_causal(const bf16* x, const bf16* w, bf16* state, bf16* y,
                       int seq, int C, int K, cudaStream_t s = 0);
    // 单 token 版本 (decode): x: [C]
    void conv1d_causal_step(const bf16* x, const bf16* w, bf16* state, bf16* y,
                            int C, int K, cudaStream_t s = 0);

    // ---------- q/k 逐头 l2 归一化 (原地) ----------
    // q: 起于列 0, k: 起于列 off; 布局 [n, stride] 行主序 (stride = qkv 通道数 C),
    // q 每 token 占用 [0, nkv*D), k 占用 [off, off+nkv*D)。q 额外乘 1/sqrt(D)
    void l2norm_qk(bf16* q, bf16* k, int n, int H, int D, int off, int stride,
                   cudaStream_t s = 0);

    // ---------- Gated Delta Rule (线性注意力) ----------
    // q,k,v: [n, stride] 行主序交错 (q 列 0, k 列 off, v 列 2*off, off = H*D),
    // beta,g: [seq, H], state: [H,D,D] fp32 (in/out)
    // out: [seq, H, D]
    // state = state * exp(g); kv = sum_k state[:,k,:]*k[k]; delta = (v-kv)*beta; state += outer(k,delta); out = sum_k state[:,k,:]*q[k]
    void gated_delta(const bf16* q, const bf16* k, const bf16* v, const bf16* beta,
                     const bf16* g, float* state, bf16* out, int seq, int H, int D,
                     int stride, cudaStream_t s = 0);

    // ---------- 全注意力 (GQA, 因果) ----------
    // q: [nq, Hq*D], kc/vc: [len, Hkv*D], out: [nq, Hq*D]
    // base = cache 已有长度 (prefill 时当前段从 cache 位置 base+qrow 起, 因果限制到 base+qrow)
    void attn_forward(const bf16* q, const bf16* kc, const bf16* vc, bf16* out,
                      int nq, int len, int Hq, int Hkv, int D, int base, bool causal,
                      float scale, cudaStream_t s = 0);

    // ---------- 视觉注意力 (非因果, 分段) ----------
    // q/k/v: [seq, stride] 行主序交错 (q 列 0, k 列 H*D, v 列 2*H*D),
    // cu_seqlens: [nseg+1] 段边界, out: [seq, H*D]
    void attn_vision(const bf16* q, const bf16* k, const bf16* v, bf16* out,
                     int seq, int H, int D, const int* cu_seqlens, int nseg,
                     float scale, int stride, cudaStream_t s = 0);

    // ---------- 视觉 patch embed (Conv3d) ----------
    // img: [C, T, H, W] fp32, w: [O, C, T, ph, pw] bf16, bias: [O] bf16
    // out: [npatches, O] bf16, order: [npatches] 块主序打散索引
    void patch_embed(const float* img, const bf16* w, const bf16* bias, bf16* out,
                     const int* order, int C, int T, int H, int W, int ph, int pw,
                     int O, cudaStream_t s = 0);

    // ---------- 位置嵌入双线性插值 ----------
    // table: [2304, D] bf16, idx: [np, 4], wgt: [np, 4] fp32 -> out: [np, D]
    void pos_embed_interp(const bf16* table, const int* idx, const float* wgt,
                          bf16* out, int np, int D, cudaStream_t s = 0);

    // ---------- LM head + argmax ----------
    // x: [D] bf16, embed: [V, D] bf16 -> out: [V] bf16 (logits)
    void lm_head(const bf16* x, const bf16* embed, bf16* out, int V, int D,
                 cudaStream_t s = 0);
    // argmax over [n] bf16 logits
    int argmax(const bf16* x, int n, cudaStream_t s = 0);
}
