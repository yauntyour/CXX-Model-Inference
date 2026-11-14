#pragma once
// ============================================================
// 通用 Dense Transformer 文本模型 (Llama 风格, BF16 推理)
//
// 面向 AstrAI-V1-2B-instruct 这类配置:
//   dense FFN + GQA + 标准 RMSNorm + 标准 RoPE (NeoX 半维配对)
// 与 Qwen3.5 的 Gated DeltaNet / mRoPE 混合架构完全独立。
// ============================================================

#include "kernels.hpp"
#include "TensorN.hpp"

#include <vector>
#include <string>
#include <deque>
#include <cstdint>

struct DenseConfig
{
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    int max_position_embeddings = 32768;
    int rot_dim = 0;                    // 旋转维数 (<= head_dim)
    bool tie_word_embeddings = false;
    bool rope_interleaved = false;      // false=Llama 半维配对, true=GPT-J 交错
    bool norm_1pw = false;              // false=y*rms*w, true=y*rms*(1+w)

    // 兼容两种 config 写法:
    //  1) AstrAI: attention: {num_heads, num_kv_heads, gqa:{head_dim, rotary_dim}}
    //  2) Llama : 顶层 num_attention_heads / num_key_value_heads / head_dim
    bool load(const std::string& config_path);
};

struct QuantW
{
    TensorN::CudaTensor<int8_t> q;      // [N, K] 行主序
    TensorN::CudaTensor<float> scale;   // [N]
};

struct DenseAttnW
{
    // 保持 PyTorch 原始布局 [out, in], 不再做 CPU 转置:
    // qkv_w: [(Hq + 2*Hkv)*D, hidden], 行块 [Q | K | V]
    // o_w:   [hidden, Hq*D]
    TensorN::CudaTensor<TensorN::bfloat16> qkv_w, o_w;
    QuantW qkv_q, o_q;                  // W8A16 decode
};

struct DenseMLPW
{
    // gu_w: [2*I, hidden] 行块 [gate | up]
    // down_w: [hidden, I]
    TensorN::CudaTensor<TensorN::bfloat16> gu_w, down_w;
    QuantW gu_q, down_q;                // W8A16 decode
};

struct DenseLayerW
{
    TensorN::CudaTensor<TensorN::bfloat16> in_norm;              // [hidden]
    TensorN::CudaTensor<TensorN::bfloat16> post_norm;            // [hidden]
    DenseAttnW attn;
    DenseMLPW mlp;
};

struct DenseKVState
{
    TensorN::CudaTensor<TensorN::bfloat16> k, v;   // [max_ctx, Hkv * D]
};

class DenseModel
{
public:
    DenseConfig cfg;

    TensorN::CudaTensor<TensorN::bfloat16> embed;       // [V, hidden]
    TensorN::CudaTensor<TensorN::bfloat16> lm_head_w;   // [V, hidden] (可与 embed 不同)
    QuantW lm_head_q;                                   // W8A16 lm_head
    TensorN::CudaTensor<TensorN::bfloat16> final_norm;  // [hidden]
    std::vector<DenseLayerW> layers;
    std::vector<DenseKVState> kv;
    TensorN::CudaTensor<float> d_inv_freq;              // [rot_dim / 2]

    int max_ctx = 8192;         // KV cache 容量 (在 load 前设置)
    int past_len = 0;
    bool tied_lm_head = false;  // 无 lm_head.weight 时复用 embed
    bool fast_ops = true;       // 快路径: GEMV / 融合 QKV-RoPE / warp 注意力
    bool use_w8a16 = true;      // decode 使用 int8 权重 + bf16 激活
    bool skip_attn = false;     // 调试: 跳过注意力, 用于定位 decode 瓶颈
    bool debug_timing = false;
    bool debug_logits = false;  // prefill 后打印 top-10 logits

    bool load(const std::string& model_dir, const std::string& safetensors_path);

    // 前向: ids [S] (host); 位置从 past_len 递增; 返回最后 token 的 greedy id
    int forward(const std::vector<int>& ids);

    void reset_state();

private:
    // 工作缓冲 (按 S 懒分配)
    TensorN::CudaTensor<TensorN::bfloat16> b_x, b_h, b_q, b_k, b_v, b_attn;
    TensorN::CudaTensor<TensorN::bfloat16> b_out, b_gate, b_up;
    TensorN::CudaTensor<TensorN::bfloat16> b_qkv, b_gu, b_hid;
    TensorN::CudaTensor<TensorN::bfloat16> b_logits;
    TensorN::CudaTensor<int> d_ids, d_pos;
    int buf_s = 0;

    void ensure_buffers(int S);
    void run_layer(int li, int S);
};
