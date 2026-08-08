#pragma once
// ============================================================
// Qwen3.5-0.8B 文本模型 (TensorN + 自定义 CUDA 内核, BF16 推理)
// ============================================================

#include "kernels.hpp"
#include "TensorN.hpp"

#include <vector>
#include <string>
#include <memory>

struct QwenConfig
{
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int linear_num_key_heads = 0;
    int linear_num_value_heads = 0;
    int linear_key_head_dim = 0;
    int linear_value_head_dim = 0;
    int linear_conv_kernel_dim = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000000.0f;
    float partial_rotary_factor = 1.0f;
    int rot_dim = 0;                    // head_dim * partial_rotary_factor
    int mrope_n_freq = 0;               // rot_dim / 2
    std::vector<int> mrope_section;     // {11, 11, 10}
    std::vector<int> layer_types;       // 0 = linear_attention, 1 = full_attention
    int eos_token_id = 0;
    int image_token_id = 0;

    bool load(const std::string& config_path);
};

// 线性注意力层权重 (全部 BF16, 行主序转置为 [in, out] 供 matmul)
struct LinearAttnW
{
    TensorN::CudaTensor<TensorN::bfloat16> qkv_t, z_t, b_t, a_t, out_t;  // 转置权重
    TensorN::CudaTensor<TensorN::bfloat16> conv_w;                        // [C, K] 深度卷积
    TensorN::CudaTensor<TensorN::bfloat16> norm_w;                        // [head_v_dim]
    TensorN::CudaTensor<float> A_log;                                     // [num_v_heads] fp32
    TensorN::CudaTensor<TensorN::bfloat16> dt_bias;                       // [num_v_heads] bf16
};

struct FullAttnW
{
    TensorN::CudaTensor<TensorN::bfloat16> q_t, k_t, v_t, o_t;            // 转置权重
    TensorN::CudaTensor<TensorN::bfloat16> q_norm, k_norm;                // [head_dim]
};

struct MLPW
{
    TensorN::CudaTensor<TensorN::bfloat16> gate_t, up_t, down_t;
};

struct LayerW
{
    int type = 0;
    TensorN::CudaTensor<TensorN::bfloat16> in_ln, post_ln;                // [hidden]
    std::unique_ptr<LinearAttnW> la;
    std::unique_ptr<FullAttnW> fa;
    MLPW mlp;
};

struct KVState
{
    TensorN::CudaTensor<TensorN::bfloat16> k, v;   // [max_len, Hkv, D]
    int len = 0;
};

class QwenModel
{
public:
    QwenConfig cfg;
    TensorN::CudaTensor<TensorN::bfloat16> embed;                 // [V, hidden] 词表(兼 lm_head)
    std::vector<LayerW> layers;
    TensorN::CudaTensor<TensorN::bfloat16> final_norm;
    TensorN::CudaTensor<float> d_inv_freq;               // [mrope_n_freq]
    TensorN::CudaTensor<int> d_section;                  // [3]
    std::vector<KVState> kv;                             // 每个 full 层
    TensorN::CudaTensor<TensorN::bfloat16> conv_state;            // [n_linear, C, K-1]
    TensorN::CudaTensor<float> rec_state;                // [n_linear, H, D, D] fp32

    int n_linear = 0;
    std::vector<int> linear_layer_idx;                   // 线性层在 layers 中的下标
    std::vector<int> full_layer_idx;                     // full 层下标

    int past_len = 0;            // KV cache 总长度
    int rope_delta = 0;          // mrope 增量 (生成时加到位置)

    bool debug_timing = false;   // prefill 时打印每阶段耗时

    bool load(const std::string& model_dir, const std::string& safetensors_path);

    // 前向: ids [S] (host), 可选图片特征 (device, [N, hidden]) 覆盖到 img_at 起的连续位置
    // pos3: 3*S 个位置 (T/H/W); 返回最后一个 token 的 greedy 采样 id
    int forward(const std::vector<int>& ids,
                const TensorN::CudaTensor<TensorN::bfloat16>* img_feat, int img_at,
                const std::vector<int>& pos3);

    // 重置 KV / 线性状态 (新一轮)
    void reset_state();

private:
    // 工作缓冲区 (懒分配)
    TensorN::CudaTensor<TensorN::bfloat16> b_emb, b_h, b_x;
    TensorN::CudaTensor<TensorN::bfloat16> b_qkv, b_qkv2, b_z, b_a, b_b, b_attn, b_gate, b_qq, b_q, b_kv, b_kv2, b_mlp;
    TensorN::CudaTensor<TensorN::bfloat16> b_out, b_up;   // [S, H] / [S, I] 专用输出缓冲
    TensorN::CudaTensor<TensorN::bfloat16> b_logits;
    TensorN::CudaTensor<int> d_ids, d_pos_t, d_pos_h, d_pos_w;

    void ensure_size();
    void run_layer(int li, int S, int kv_base);
    void run_linear(LayerW& L, int li, int S);
    void run_full(LayerW& L, int li, int S, int kv_base);
    void run_mlp(LayerW& L, int li, int S);
};
