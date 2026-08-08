#pragma once
// ============================================================
// Qwen3.5 视觉编码器 (BF16, 自定义 CUDA 内核)
// ============================================================

#include "kernels.hpp"
#include "TensorN.hpp"

#include <vector>
#include <string>

// 图像预处理结果 (主机端计算)
struct VisionPrep
{
    int grid_h = 0, grid_w = 0;        // 特征网格 (resized/16)
    int n_patches = 0;                 // grid_h * grid_w
    int n_tokens = 0;                  // n_patches / 4 (merge 后)
    std::vector<float> pixels;         // [3, 2, H, W] fp32 归一化 (双帧)
    std::vector<int> interp_idx;       // [n_patches, 4] pos_embed 双线性插值索引
    std::vector<float> interp_wgt;     // [n_patches, 4]
    std::vector<int> patch_order;      // [n_patches] 块主序打散
    std::vector<int> vis_pos;          // [n_patches, 2] 视觉 rope 位置 (h, w)
};

class VisionModel
{
public:
    int depth = 12, hidden = 768, heads = 12, head_dim = 64;
    int intermediate = 3072, out_hidden = 1024;
    int patch = 16, temporal = 2, num_pos_emb = 2304, spatial_merge = 2;
    float eps = 1e-6f;

    // 权重
    TensorN::CudaTensor<TensorN::bfloat16> patch_w;   // [O, C, T, ph, pw]
    TensorN::CudaTensor<TensorN::bfloat16> patch_b;   // [O]
    TensorN::CudaTensor<TensorN::bfloat16> pos_embed; // [2304, 768]
    TensorN::CudaTensor<float> d_inv_freq;            // [16] 视觉 rope

    struct BlockW
    {
        TensorN::CudaTensor<TensorN::bfloat16> n1w, n1b, n2w, n2b;   // LayerNorm
        TensorN::CudaTensor<TensorN::bfloat16> qkv_t, qkv_b;         // [768, 2304] + bias
        TensorN::CudaTensor<TensorN::bfloat16> proj_t, proj_b;       // [768, 768] + b
        TensorN::CudaTensor<TensorN::bfloat16> fc1_t, fc1_b;         // [768, 3072] + b
        TensorN::CudaTensor<TensorN::bfloat16> fc2_t, fc2_b;         // [3072, 768] + b
    };
    std::vector<BlockW> blocks;

    struct MergerW
    {
        TensorN::CudaTensor<TensorN::bfloat16> nw, nb;
        TensorN::CudaTensor<TensorN::bfloat16> fc1_t, fc1_b;  // [768, 3072] 输入 768*4
        TensorN::CudaTensor<TensorN::bfloat16> fc2_t, fc2_b;  // [3072, 1024]
    };
    MergerW merger;

    bool load(const std::string& model_dir, const std::string& safetensors_path);

    // 前向: prep (主机) -> 视觉特征 [n_tokens, 1024] bf16 (GPU)
    TensorN::CudaTensor<TensorN::bfloat16> forward(const VisionPrep& prep);

private:
    void ensure_size(int P);
    TensorN::CudaTensor<TensorN::bfloat16> b_patches, b_pos, b_tmp, b_qkv, b_attn, b_fc;
};
