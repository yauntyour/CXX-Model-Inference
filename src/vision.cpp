// ============================================================
// Qwen3.5 视觉编码器实现
// ============================================================

#include "vision.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <cstring>
#include <cmath>
#include <stdexcept>

using namespace TensorN;

typedef TensorN::bfloat16 B16;

namespace
{
    inline qk::bf16* bp(CudaTensor<B16>& t) { return reinterpret_cast<qk::bf16*>(t.device_ptr()); }
    inline const qk::bf16* bp(const CudaTensor<B16>& t)
    {
        return reinterpret_cast<const qk::bf16*>(t.device_ptr());
    }

    std::vector<qk::bf16> to_host_bf16(const SafeTensor& st)
    {
        if (st.dtype != "BF16") throw std::runtime_error("expected BF16, got " + st.dtype);
        std::vector<qk::bf16> out(st.data.size() / 2);
        std::memcpy(out.data(), st.data.data(), st.data.size());
        return out;
    }

    std::vector<qk::bf16> transpose_bf16(const SafeTensor& st)
    {
        auto v = to_host_bf16(st);
        size_t out = st.shape.size() >= 2 ? (size_t)st.shape[0] : 1;
        size_t in = st.shape.size() >= 2 ? (size_t)st.shape[1] : (size_t)st.shape[0];
        std::vector<qk::bf16> tv(v.size());
        for (size_t i = 0; i < out; i++)
            for (size_t j = 0; j < in; j++)
                tv[j * out + i] = v[i * in + j];
        return tv;
    }

    CudaTensor<B16> load_mat(const std::unordered_map<std::string, SafeTensor>& raw,
                             const std::string& name, bool transpose)
    {
        auto it = raw.find(name);
        if (it == raw.end()) throw std::runtime_error("missing weight: " + name);
        auto& st = it->second;
        std::vector<qk::bf16> v = transpose ? transpose_bf16(st) : to_host_bf16(st);
        std::vector<size_t> sh(st.shape.begin(), st.shape.end());
        if (transpose && sh.size() >= 2) sh = {sh[1], sh[0]};
        CudaTensor<B16> t(sh);
        t.copyFromHost(reinterpret_cast<const B16*>(v.data()), v.size());
        return t;
    }

    CudaTensor<B16> load_vec(const std::unordered_map<std::string, SafeTensor>& raw,
                             const std::string& name)
    {
        auto it = raw.find(name);
        if (it == raw.end()) throw std::runtime_error("missing weight: " + name);
        auto v = to_host_bf16(it->second);
        CudaTensor<B16> t({v.size()});
        t.copyFromHost(reinterpret_cast<const B16*>(v.data()), v.size());
        return t;
    }
}

bool VisionModel::load(const std::string& model_dir, const std::string& safetensors_path)
{
    std::ifstream f(model_dir + "/config.json");
    if (!f.is_open()) return false;
    nlohmann::json j;
    f >> j;
    auto& vc = j["vision_config"];
    depth = vc["depth"];
    hidden = vc["hidden_size"];
    heads = vc["num_heads"];
    intermediate = vc["intermediate_size"];
    out_hidden = vc["out_hidden_size"];
    patch = vc["patch_size"];
    temporal = vc["temporal_patch_size"];
    num_pos_emb = vc["num_position_embeddings"];
    spatial_merge = vc["spatial_merge_size"];
    head_dim = hidden / heads;

    auto raw = load_safetensors_raw(safetensors_path);
    const std::string P = "model.visual.";

    patch_w = load_mat(raw, P + "patch_embed.proj.weight", false);   // [768,3,2,16,16]
    patch_b = load_vec(raw, P + "patch_embed.proj.bias");
    pos_embed = load_mat(raw, P + "pos_embed.weight", false);        // [2304, 768]

    std::vector<float> inv_freq(16);
    for (int i = 0; i < 16; i++)
        inv_freq[i] = 1.0f / powf(10000.0f, (float)i / 16.0f);
    d_inv_freq = CudaTensor<float>({16});
    d_inv_freq.copyFromHost(inv_freq.data(), 16);

    blocks.resize(depth);
    for (int i = 0; i < depth; i++)
    {
        std::string b = P + "blocks." + std::to_string(i) + ".";
        BlockW& w = blocks[i];
        w.n1w = load_vec(raw, b + "norm1.weight");
        w.n1b = load_vec(raw, b + "norm1.bias");
        w.n2w = load_vec(raw, b + "norm2.weight");
        w.n2b = load_vec(raw, b + "norm2.bias");
        w.qkv_t = load_mat(raw, b + "attn.qkv.weight", true);
        w.qkv_b = load_vec(raw, b + "attn.qkv.bias");
        w.proj_t = load_mat(raw, b + "attn.proj.weight", true);
        w.proj_b = load_vec(raw, b + "attn.proj.bias");
        w.fc1_t = load_mat(raw, b + "mlp.linear_fc1.weight", true);
        w.fc1_b = load_vec(raw, b + "mlp.linear_fc1.bias");
        w.fc2_t = load_mat(raw, b + "mlp.linear_fc2.weight", true);
        w.fc2_b = load_vec(raw, b + "mlp.linear_fc2.bias");
    }
    merger.nw = load_vec(raw, P + "merger.norm.weight");
    merger.nb = load_vec(raw, P + "merger.norm.bias");
    merger.fc1_t = load_mat(raw, P + "merger.linear_fc1.weight", true);
    merger.fc1_b = load_vec(raw, P + "merger.linear_fc1.bias");
    merger.fc2_t = load_mat(raw, P + "merger.linear_fc2.weight", true);
    merger.fc2_b = load_vec(raw, P + "merger.linear_fc2.bias");
    return true;
}

void VisionModel::ensure_size(int P)
{
    if (b_patches.size() < (size_t)P * hidden)
    {
        b_patches = CudaTensor<B16>({(size_t)P, (size_t)hidden});
        b_pos = CudaTensor<B16>({(size_t)P, (size_t)hidden});
        b_tmp = CudaTensor<B16>({(size_t)P, (size_t)hidden});
        b_qkv = CudaTensor<B16>({(size_t)P, (size_t)(hidden * 3)});
        b_attn = CudaTensor<B16>({(size_t)P, (size_t)hidden});
        b_fc = CudaTensor<B16>({(size_t)P, (size_t)intermediate});
    }
}

CudaTensor<B16> VisionModel::forward(const VisionPrep& prep)
{
    const int P = prep.n_patches;
    const int H = hidden;
    ensure_size(P);

    // 上传预处理数据
    CudaTensor<float> d_img({3, 2, (size_t)prep.grid_h * patch, (size_t)prep.grid_w * patch});
    d_img.copyFromHost(prep.pixels.data(), prep.pixels.size());
    CudaTensor<int> d_order({(size_t)P});
    d_order.copyFromHost(prep.patch_order.data(), P);
    CudaTensor<int> d_idx({(size_t)P, 4});
    d_idx.copyFromHost(prep.interp_idx.data(), (size_t)P * 4);
    CudaTensor<float> d_wgt({(size_t)P, 4});
    d_wgt.copyFromHost(prep.interp_wgt.data(), (size_t)P * 4);
    CudaTensor<int> d_pos({(size_t)P, 2});
    d_pos.copyFromHost(prep.vis_pos.data(), (size_t)P * 2);
    CudaTensor<int> d_cu({2});
    int cu[2] = {0, P};
    d_cu.copyFromHost(cu, 2);

    // patch embed + 位置嵌入
    qk::patch_embed(d_img.device_ptr(), bp(patch_w), bp(patch_b), bp(b_patches),
                    d_order.device_ptr(), 3, temporal, prep.grid_h * patch, prep.grid_w * patch,
                    patch, patch, H);
    qk::pos_embed_interp(bp(pos_embed), d_idx.device_ptr(), d_wgt.device_ptr(),
                         bp(b_pos), P, H);
    qk::add_inplace(bp(b_patches), bp(b_pos), P * H);

    // 12 个 block
    for (int i = 0; i < depth; i++)
    {
        BlockW& w = blocks[i];
        // attention
        qk::layer_norm(bp(b_patches), bp(w.n1w), bp(w.n1b), bp(b_tmp), P, H, eps);
        cuda::matmul(b_tmp, w.qkv_t, b_qkv);
        qk::add_bias(bp(b_qkv), bp(w.qkv_b), bp(b_qkv), P, hidden * 3);
        // b_qkv 为 [P, 3*hidden] 行主序, 每 token 内 [q | k | v] (各 H*D);
        // rope 按行步长 3*hidden 原地写 (每位置仅一个线程读写, 无竞态)
        qk::bf16* qb = bp(b_qkv);
        qk::bf16* kb = qb + (size_t)hidden;
        qk::bf16* vb = qb + 2 * (size_t)hidden;
        const int vstride = 3 * hidden;      // 每 token [q|k|v] 总宽
        qk::rope_vision(qb, qb, d_pos.device_ptr(), d_inv_freq.device_ptr(),
                        P, heads, head_dim, vstride, vstride);
        qk::rope_vision(kb, kb, d_pos.device_ptr(), d_inv_freq.device_ptr(),
                        P, heads, head_dim, vstride, vstride);
        qk::attn_vision(qb, kb, vb, bp(b_attn),
                        P, heads, head_dim, d_cu.device_ptr(), 1, 1.0f / sqrtf((float)head_dim),
                        vstride);
        cuda::matmul(b_attn, w.proj_t, b_tmp);
        qk::add_bias(bp(b_tmp), bp(w.proj_b), bp(b_tmp), P, H);
        qk::add_inplace(bp(b_patches), bp(b_tmp), P * H);
        // mlp
        qk::layer_norm(bp(b_patches), bp(w.n2w), bp(w.n2b), bp(b_tmp), P, H, eps);
        cuda::matmul(b_tmp, w.fc1_t, b_fc);
        qk::add_bias(bp(b_fc), bp(w.fc1_b), bp(b_fc), P, intermediate);
        qk::gelu_tanh(bp(b_fc), bp(b_fc), P * intermediate);
        cuda::matmul(b_fc, w.fc2_t, b_tmp);
        qk::add_bias(bp(b_tmp), bp(w.fc2_b), bp(b_tmp), P, H);
        qk::add_inplace(bp(b_patches), bp(b_tmp), P * H);
    }

    // merger (与 HF 一致): 先按 patch (768) 逐行 LayerNorm, 再按 2x2 块拼成 [nt, 3072],
    // fc1 -> 精确 GELU -> fc2 -> [nt, 1024]
    const int nt = prep.n_tokens;
    const int Dm = hidden * spatial_merge * spatial_merge;   // 3072
    qk::layer_norm(bp(b_patches), bp(merger.nw), bp(merger.nb), bp(b_tmp), P, H, eps);
    auto in_view = b_tmp.reshape({(size_t)nt, (size_t)Dm});
    CudaTensor<B16> d_fc1({(size_t)nt, (size_t)Dm});
    cuda::matmul(in_view, merger.fc1_t, d_fc1);
    qk::add_bias(bp(d_fc1), bp(merger.fc1_b), bp(d_fc1), nt, Dm);
    qk::gelu(bp(d_fc1), bp(d_fc1), nt * Dm);
    CudaTensor<B16> d_merged({(size_t)nt, (size_t)out_hidden});
    cuda::matmul(d_fc1, merger.fc2_t, d_merged);
    qk::add_bias(bp(d_merged), bp(merger.fc2_b), bp(d_merged), nt, out_hidden);
    return d_merged;
}
