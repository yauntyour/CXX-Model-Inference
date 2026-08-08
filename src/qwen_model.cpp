// ============================================================
// Qwen3.5-0.8B 文本模型实现 (BF16 + fp32 累加, 自定义 CUDA 内核)
// ============================================================

#include "qwen_model.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <stdexcept>

using namespace TensorN;

typedef TensorN::bfloat16 B16;

namespace vw { inline CudaTensor<B16> view2(CudaTensor<B16>& t, size_t m, size_t n) { return t.reshape({m, n}); } }

namespace
{
    // 检查粘滞的 CUDA 错误 (不同步, 仅查询)
    inline void chk_err(const char* where)
    {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("CUDA error at ") + where + ": " +
                                     cudaGetErrorString(e));
    }

    class StageTimer
    {
    public:
        StageTimer(bool on, const char* name) : enabled(on), label(name)
        {
            if (enabled) t0 = std::chrono::high_resolution_clock::now();
        }
        ~StageTimer()
        {
            if (!enabled) return;
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "  [" << label << " " << (int)ms << " ms]" << std::endl;
        }
        bool enabled;
        const char* label;
        std::chrono::high_resolution_clock::time_point t0;
    };

    static bool g_dump = false;

    // 调试: dump 设备端 bf16 向量到文件
    inline void dump_vec(const char* name, const qk::bf16* p, int n)
    {
        FILE* f = fopen(name, "wb");
        if (!f) return;
        std::vector<B16> v(n);
        cudaMemcpy(v.data(), p, (size_t)n * 2, cudaMemcpyDeviceToHost);
        fwrite(v.data(), 2, n, f);
        fclose(f);
    }

    inline qk::bf16* bp(CudaTensor<B16>& t) { return reinterpret_cast<qk::bf16*>(t.device_ptr()); }
    inline const qk::bf16* bp(const CudaTensor<B16>& t)
    {
        return reinterpret_cast<const qk::bf16*>(t.device_ptr());
    }

    void check(const SafeTensor* st, const std::string& name)
    {
        if (!st) throw std::runtime_error("missing weight: " + name);
    }

    // BF16 原始字节 -> 主机 bf16 向量
    std::vector<qk::bf16> to_host_bf16(const SafeTensor& st)
    {
        if (st.dtype != "BF16") throw std::runtime_error("expected BF16, got " + st.dtype);
        size_t n = st.data.size() / 2;
        std::vector<qk::bf16> out(n);
        std::memcpy(out.data(), st.data.data(), st.data.size());
        return out;
    }

    // 转置权重: [out, in] -> [in, out]
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

    // 线性层权重 [out, in] -> 转置 [in, out]
    CudaTensor<B16> mat_t(const std::unordered_map<std::string, SafeTensor>& raw,
                          const std::string& name)
    {
        auto it = raw.find(name);
        check(it != raw.end() ? &it->second : nullptr, name);
        auto v = transpose_bf16(it->second);
        size_t in = it->second.shape.size() >= 2 ? (size_t)it->second.shape[1] : 1;
        size_t out = it->second.shape.size() >= 2 ? (size_t)it->second.shape[0] : 1;
        CudaTensor<B16> t({in, out});
        t.copyFromHost(reinterpret_cast<const B16*>(v.data()), v.size());
        return t;
    }

    // 原始取向 [out, in] (embed 词表)
    CudaTensor<B16> mat_plain(const std::unordered_map<std::string, SafeTensor>& raw,
                              const std::string& name)
    {
        auto it = raw.find(name);
        check(it != raw.end() ? &it->second : nullptr, name);
        auto v = to_host_bf16(it->second);
        std::vector<size_t> sh(it->second.shape.begin(), it->second.shape.end());
        CudaTensor<B16> t(sh);
        t.copyFromHost(reinterpret_cast<const B16*>(v.data()), v.size());
        return t;
    }

    // 1D 权重 (norm 等), 不转置; 支持 BF16 / F32
    CudaTensor<B16> vec1(const std::unordered_map<std::string, SafeTensor>& raw,
                         const std::string& name)
    {
        auto it = raw.find(name);
        check(it != raw.end() ? &it->second : nullptr, name);
        auto& st = it->second;
        std::vector<qk::bf16> v;
        if (st.dtype == "BF16")
        {
            v.resize(st.data.size() / 2);
            std::memcpy(v.data(), st.data.data(), st.data.size());
        }
        else if (st.dtype == "F32")
        {
            size_t n = st.data.size() / 4;
            const float* p = reinterpret_cast<const float*>(st.data.data());
            v.resize(n);
            for (size_t i = 0; i < n; i++) v[i] = qk::bf16(p[i]);
        }
        else throw std::runtime_error("vec1: unsupported dtype " + st.dtype);
        CudaTensor<B16> t({v.size()});
        t.copyFromHost(reinterpret_cast<const B16*>(v.data()), v.size());
        return t;
    }

    // FP32 1D 权重
    CudaTensor<float> vec_f32(const std::unordered_map<std::string, SafeTensor>& raw,
                              const std::string& name)
    {
        auto it = raw.find(name);
        check(it != raw.end() ? &it->second : nullptr, name);
        auto& st = it->second;
        std::vector<float> v(st.data.size() / 4);
        std::memcpy(v.data(), st.data.data(), st.data.size());
        CudaTensor<float> t({v.size()});
        t.copyFromHost(v.data(), v.size());
        return t;
    }

    // BF16 1D -> fp32 host
    std::vector<float> to_host_f32(const SafeTensor& st)
    {
        auto v = to_host_bf16(st);
        std::vector<float> out(v.size());
        for (size_t i = 0; i < v.size(); i++)
            out[i] = fp::bfloat16_bits_to_float(*reinterpret_cast<const uint16_t*>(&v[i]));
        return out;
    }
} // namespace

bool QwenConfig::load(const std::string& config_path)
{
    std::ifstream f(config_path);
    if (!f.is_open()) return false;
    nlohmann::json j;
    f >> j;
    auto& tc = j["text_config"];
    hidden_size = tc["hidden_size"];
    intermediate_size = tc["intermediate_size"];
    num_hidden_layers = tc["num_hidden_layers"];
    num_attention_heads = tc["num_attention_heads"];
    num_key_value_heads = tc["num_key_value_heads"];
    head_dim = tc.value("head_dim", hidden_size / num_attention_heads);
    vocab_size = tc["vocab_size"];
    linear_num_key_heads = tc["linear_num_key_heads"];
    linear_num_value_heads = tc["linear_num_value_heads"];
    linear_key_head_dim = tc["linear_key_head_dim"];
    linear_value_head_dim = tc["linear_value_head_dim"];
    linear_conv_kernel_dim = tc["linear_conv_kernel_dim"];
    rms_norm_eps = tc["rms_norm_eps"];
    auto& rp = tc["rope_parameters"];
    rope_theta = rp["rope_theta"];
    partial_rotary_factor = rp.value("partial_rotary_factor", 1.0f);
    for (auto& s : rp["mrope_section"]) mrope_section.push_back(s);
    for (auto& lt : tc["layer_types"])
        layer_types.push_back(lt == "full_attention" ? 1 : 0);
    eos_token_id = tc["eos_token_id"];
    image_token_id = j.value("image_token_id", 0);
    rot_dim = (int)(head_dim * partial_rotary_factor);
    mrope_n_freq = rot_dim / 2;
    return true;
}

bool QwenModel::load(const std::string& model_dir, const std::string& safetensors_path)
{
    if (!cfg.load(model_dir + "/config.json"))
        return false;

    auto raw = load_safetensors_raw(safetensors_path);
    const std::string P = "model.language_model.";

    // 词表 (不转置: gather 与 lm_head 均按行读)
    embed = mat_plain(raw, P + "embed_tokens.weight");

    // 层
    int la_idx = 0;  // 线性层序号 (状态数组索引)
    layers.resize(cfg.num_hidden_layers);
    for (int i = 0; i < cfg.num_hidden_layers; i++)
    {
        LayerW& L = layers[i];
        L.type = cfg.layer_types[i];
        L.in_ln = vec1(raw, P + "layers." + std::to_string(i) + ".input_layernorm.weight");
        L.post_ln = vec1(raw, P + "layers." + std::to_string(i) + ".post_attention_layernorm.weight");
        L.mlp.gate_t = mat_t(raw, P + "layers." + std::to_string(i) + ".mlp.gate_proj.weight");
        L.mlp.up_t = mat_t(raw, P + "layers." + std::to_string(i) + ".mlp.up_proj.weight");
        L.mlp.down_t = mat_t(raw, P + "layers." + std::to_string(i) + ".mlp.down_proj.weight");

        if (L.type == 0)
        {
            std::string base = P + "layers." + std::to_string(i) + ".linear_attn.";
            L.la.reset(new LinearAttnW());
            L.la->qkv_t = mat_t(raw, base + "in_proj_qkv.weight");
            L.la->z_t = mat_t(raw, base + "in_proj_z.weight");
            L.la->b_t = mat_t(raw, base + "in_proj_b.weight");
            L.la->a_t = mat_t(raw, base + "in_proj_a.weight");
            L.la->out_t = mat_t(raw, base + "out_proj.weight");
            // conv1d: [C, 1, K] -> [C, K] (连续)
            {
                auto& st = raw.at(base + "conv1d.weight");
                auto v = to_host_bf16(st);
                size_t C = (size_t)st.shape[0], K = (size_t)st.shape[2];
                CudaTensor<B16> t({C, K});
                t.copyFromHost(reinterpret_cast<const B16*>(v.data()), v.size());
                L.la->conv_w = std::move(t);
            }
            L.la->norm_w = vec1(raw, base + "norm.weight");
            // A_log 文件内 F32 (HF 以 fp32 参与计算), dt_bias 为 bf16
            {
                auto& st = raw.at(base + "A_log");
                std::vector<float> vf(st.data.size() / 4);
                std::memcpy(vf.data(), st.data.data(), st.data.size());
                CudaTensor<float> t({vf.size()});
                t.copyFromHost(vf.data(), vf.size());
                L.la->A_log = std::move(t);
            }
            {
                auto& st = raw.at(base + "dt_bias");
                auto v = to_host_bf16(st);
                CudaTensor<B16> t({v.size()});
                t.copyFromHost(reinterpret_cast<const B16*>(v.data()), v.size());
                L.la->dt_bias = std::move(t);
            }
            linear_layer_idx.push_back(i);
        }
        else
        {
            std::string base = P + "layers." + std::to_string(i) + ".self_attn.";
            L.fa.reset(new FullAttnW());
            L.fa->q_t = mat_t(raw, base + "q_proj.weight");
            L.fa->k_t = mat_t(raw, base + "k_proj.weight");
            L.fa->v_t = mat_t(raw, base + "v_proj.weight");
            L.fa->o_t = mat_t(raw, base + "o_proj.weight");
            L.fa->q_norm = vec1(raw, base + "q_norm.weight");
            L.fa->k_norm = vec1(raw, base + "k_norm.weight");
            full_layer_idx.push_back(i);
        }
    }
    n_linear = (int)linear_layer_idx.size();

    final_norm = vec1(raw, P + "norm.weight");

    // RoPE
    std::vector<float> inv_freq(cfg.mrope_n_freq);
    for (int i = 0; i < cfg.mrope_n_freq; i++)
        inv_freq[i] = 1.0f / powf(cfg.rope_theta, (float)i / cfg.mrope_n_freq);
    d_inv_freq = CudaTensor<float>({(size_t)cfg.mrope_n_freq});
    d_inv_freq.copyFromHost(inv_freq.data(), inv_freq.size());
    std::vector<int> section = cfg.mrope_section;
    d_section = CudaTensor<int>({3});
    d_section.copyFromHost(section.data(), 3);

    // 状态
    int C = cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2 +
            cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    conv_state = CudaTensor<B16>({(size_t)n_linear, (size_t)C, (size_t)(cfg.linear_conv_kernel_dim - 1)});
    rec_state = CudaTensor<float>({(size_t)n_linear, (size_t)cfg.linear_num_value_heads,
                                   (size_t)cfg.linear_key_head_dim, (size_t)cfg.linear_value_head_dim});
    conv_state.memset_zero();
    rec_state.memset_zero();

    // KV cache: 每 full 层 [max_len, Hkv, head_dim]
    const int MAX_LEN = 4096;
    for (size_t i = 0; i < full_layer_idx.size(); i++)
    {
        KVState st;
        st.k = CudaTensor<B16>({(size_t)MAX_LEN, (size_t)cfg.num_key_value_heads, (size_t)cfg.head_dim});
        st.v = CudaTensor<B16>({(size_t)MAX_LEN, (size_t)cfg.num_key_value_heads, (size_t)cfg.head_dim});
        kv.push_back(std::move(st));
    }

    past_len = 0;
    rope_delta = 0;
    return true;
}

void QwenModel::reset_state()
{
    past_len = 0;
    rope_delta = 0;
    for (auto& s : kv) s.len = 0;
    conv_state.memset_zero();
    rec_state.memset_zero();
}

void QwenModel::ensure_size()
{
    (void)0;  // 各缓冲在需要时分配
}

int QwenModel::forward(const std::vector<int>& ids,
                       const CudaTensor<B16>* img_feat, int img_at,
                       const std::vector<int>& pos3)
{
    const int S = (int)ids.size();
    const int H = cfg.hidden_size;
    const bool dbg = debug_timing && S > 1;
    g_dump = dbg;
    if (past_len + S > 4096)
        throw std::runtime_error("context too long (max 4096)");
    if ((int)pos3.size() != 3 * S)
        throw std::runtime_error("pos3 size mismatch");

    // ---- 工作缓冲 (按需分配) ----
    size_t Sm = (size_t)S;
    if (b_x.size() < (size_t)S * H)
    {
        b_emb = CudaTensor<B16>({Sm, (size_t)H});
        b_h = CudaTensor<B16>({Sm, (size_t)H});
        b_x = CudaTensor<B16>({Sm, (size_t)H});
        b_qkv = CudaTensor<B16>({Sm, (size_t)(cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2 +
                                              cfg.linear_num_value_heads * cfg.linear_value_head_dim)});
        b_qkv2 = CudaTensor<B16>({Sm, (size_t)(cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2 +
                                               cfg.linear_num_value_heads * cfg.linear_value_head_dim)});
        b_z = CudaTensor<B16>({Sm, (size_t)(cfg.linear_num_value_heads * cfg.linear_value_head_dim)});
        b_a = CudaTensor<B16>({Sm, (size_t)cfg.linear_num_value_heads});
        b_b = CudaTensor<B16>({Sm, (size_t)cfg.linear_num_value_heads});
        b_attn = CudaTensor<B16>({Sm, (size_t)(cfg.linear_num_value_heads * cfg.linear_value_head_dim)});
        b_gate = CudaTensor<B16>({Sm, (size_t)(cfg.num_attention_heads * cfg.head_dim)});
        b_qq = CudaTensor<B16>({Sm, (size_t)(cfg.num_attention_heads * cfg.head_dim * 2)});
        b_q = CudaTensor<B16>({Sm, (size_t)(cfg.num_attention_heads * cfg.head_dim)});
        b_kv = CudaTensor<B16>({Sm, (size_t)(cfg.num_key_value_heads * cfg.head_dim)});
        b_kv2 = CudaTensor<B16>({Sm, (size_t)(cfg.num_key_value_heads * cfg.head_dim)});
        b_mlp = CudaTensor<B16>({Sm, (size_t)cfg.intermediate_size});
        b_out = CudaTensor<B16>({Sm, (size_t)H});
        b_up = CudaTensor<B16>({Sm, (size_t)cfg.intermediate_size});
    }
    if (b_logits.size() < (size_t)cfg.vocab_size)
        b_logits = CudaTensor<B16>({(size_t)cfg.vocab_size});
    // d_ids / d_pos 固定分配 4096 (上下文上限), 解码时只拷贝 S 个
    if (d_ids.size() < 4096)
    {
        d_ids = CudaTensor<int>({4096});
        d_pos_t = CudaTensor<int>({4096});
        d_pos_h = CudaTensor<int>({4096});
        d_pos_w = CudaTensor<int>({4096});
    }

    // ---- 上传输入 ----
    cudaMemcpyAsync(d_ids.device_ptr(), ids.data(), S * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_pos_t.device_ptr(), pos3.data(), S * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_pos_h.device_ptr(), pos3.data() + S, S * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_pos_w.device_ptr(), pos3.data() + 2 * S, S * sizeof(int), cudaMemcpyHostToDevice);
    chk_err("upload");

    // ---- embed ----
    qk::embed_gather(bp(embed), d_ids.device_ptr(), bp(b_emb), S, H);
    if (img_feat && img_at >= 0)
    {
        int N = (int)(img_feat->size() / H);
        cudaMemcpyAsync(bp(b_emb) + (size_t)img_at * H, bp(*img_feat),
                        (size_t)N * H * sizeof(qk::bf16), cudaMemcpyDeviceToDevice);
    }
    cudaMemcpyAsync(bp(b_x), bp(b_emb), (size_t)S * H * sizeof(qk::bf16), cudaMemcpyDeviceToDevice);
    chk_err("embed");
    if (dbg)
    {
        FILE* ef = fopen("embed_dump.bin", "wb");
        std::vector<B16> rows((size_t)S * H);
        cudaMemcpy(rows.data(), bp(b_emb), (size_t)S * H * 2, cudaMemcpyDeviceToHost);
        fwrite(rows.data(), 2, (size_t)S * H, ef);
        fclose(ef);
    }

    // ---- 层 ----
    int kv_base = past_len;
    FILE* dumpf = nullptr;
    if (dbg) dumpf = fopen("hidden_dump.bin", "wb");
    for (int li = 0; li < (int)layers.size(); li++)
    {
        if (dbg) std::cout << "  [layer " << li
                           << (layers[li].type == 0 ? " linear]" : " full  ]") << std::flush;
        {
            StageTimer st(dbg, "run");
            run_layer(li, S, kv_base);
        }
        try
        {
            chk_err("layer");
        }
        catch (const std::runtime_error& e)
        {
            throw std::runtime_error(std::string("layer ") + std::to_string(li) + ": " + e.what());
        }
        if (dumpf)
        {
            // 最后一行的 hidden state (1024 bf16) 追加写入
            std::vector<B16> row(H);
            cudaMemcpy(row.data(), bp(b_x) + (size_t)(S - 1) * H, H * 2,
                       cudaMemcpyDeviceToHost);
            fwrite(row.data(), 2, H, dumpf);
        }
    }
    if (dumpf) fclose(dumpf);

    // ---- 最终归一化 + lm head ----
    qk::rms_norm_1pw(bp(b_x), bp(final_norm), bp(b_h), S, H, cfg.rms_norm_eps);
    qk::lm_head(bp(b_h) + (size_t)(S - 1) * H, bp(embed), bp(b_logits),
                cfg.vocab_size, H);
    int tok = qk::argmax(bp(b_logits), cfg.vocab_size);
    chk_err("lm_head");

    if (dbg)
    {
        // 输出 logits top-10 (仅 prefill)
        std::vector<qk::bf16> lg(cfg.vocab_size);
        cudaMemcpy(lg.data(), bp(b_logits), (size_t)cfg.vocab_size * 2,
                   cudaMemcpyDeviceToHost);
        std::vector<std::pair<float, int>> top;
        for (int i = 0; i < cfg.vocab_size; i++)
            top.emplace_back(fp::bfloat16_bits_to_float(
                                 *reinterpret_cast<const uint16_t*>(&lg[i])), i);
        std::partial_sort(top.begin(), top.begin() + 10, top.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        std::cout << "  [logits top10]: ";
        for (int i = 0; i < 10; i++)
            std::cout << top[i].second << "(" << top[i].first << ") ";
        std::cout << std::endl;
    }

    past_len += S;
    return tok;
}

void QwenModel::run_layer(int li, int S, int kv_base)
{
    LayerW& L = layers[li];
    const int H = cfg.hidden_size;
    qk::rms_norm_1pw(bp(b_x), bp(L.in_ln), bp(b_h), S, H, cfg.rms_norm_eps);
    if (L.type == 0)
        run_linear(L, li, S);
    else
        run_full(L, li, S, kv_base);
    qk::rms_norm_1pw(bp(b_x), bp(L.post_ln), bp(b_h), S, H, cfg.rms_norm_eps);
    run_mlp(L, li, S);
}

void QwenModel::run_linear(LayerW& L, int li, int S)
{
    LinearAttnW& la = *L.la;
    const int H = cfg.hidden_size;
    const int nkv = cfg.linear_num_key_heads;   // == num_v_heads
    const int D = cfg.linear_key_head_dim;
    const int C = 2 * nkv * D + nkv * D;        // qkv 通道
    const int la_idx = (int)(std::lower_bound(linear_layer_idx.begin(), linear_layer_idx.end(), li) -
                             linear_layer_idx.begin());
    const bool dmp = g_dump && li == 0;

    cuda::matmul(b_h, la.qkv_t, b_qkv); chk_err("lin qkv");
    cuda::matmul(b_h, la.z_t, b_z); chk_err("lin z");
    cuda::matmul(b_h, la.b_t, b_b); chk_err("lin b");
    cuda::matmul(b_h, la.a_t, b_a); chk_err("lin a");
    if (dmp)
    {
        dump_vec("d_inln.bin", bp(b_h), S * H);
        dump_vec("d_qkv.bin", bp(b_qkv), S * C);
        dump_vec("d_z.bin", bp(b_z), S * nkv * D);
        dump_vec("d_b.bin", bp(b_b), S * nkv);
        dump_vec("d_a.bin", bp(b_a), S * nkv);
    }

    qk::conv1d_causal(bp(b_qkv), bp(la.conv_w), bp(conv_state) + (size_t)la_idx * C * 3,
                      bp(b_qkv2), S, C, cfg.linear_conv_kernel_dim);
    chk_err("lin conv");
    if (dmp) dump_vec("d_qkv_conv.bin", bp(b_qkv2), S * C);

    // b_qkv2 为 [S, C] 行主序, 每 token 内 [q | k | v] (各 nkv*D); 内核按行步长 C 访问
    qk::bf16* base = bp(b_qkv2);
    qk::bf16* q = base;
    qk::bf16* k = base + (size_t)nkv * D;
    qk::bf16* v = base + 2 * (size_t)nkv * D;
    const int stride = C;
    qk::l2norm_qk(q, k, S, nkv, D, nkv * D, stride);
    chk_err("lin l2norm");
    qk::sigmoid(bp(b_b), bp(b_b), S * nkv);
    qk::gated_delta_params(bp(b_a), la.A_log.device_ptr(), bp(la.dt_bias), bp(b_a), S * nkv, nkv);
    chk_err("lin params");
    if (dmp)
    {
        dump_vec("d_q_l2.bin", base, nkv * D);                    // row 0 q
        dump_vec("d_k_l2.bin", base + nkv * D, nkv * D);          // row 0 k
        dump_vec("d_beta.bin", bp(b_b), S * nkv);
        dump_vec("d_g.bin", bp(b_a), S * nkv);
    }

    qk::gated_delta(q, k, v, bp(b_b), bp(b_a),
                    rec_state.device_ptr() + (size_t)la_idx * nkv * D * D,
                    bp(b_attn), S, nkv, D, stride);
    chk_err("lin gated");
    if (dmp) dump_vec("d_gated.bin", bp(b_attn), S * nkv * D);
    qk::rms_norm_gated(bp(b_attn), bp(la.norm_w), bp(b_z), bp(b_attn), S * nkv, D,
                       cfg.rms_norm_eps);
    chk_err("lin norm_gated");
    if (dmp) dump_vec("d_normgated.bin", bp(b_attn), nkv * D);   // row 0

    cuda::matmul(b_attn, la.out_t, b_out); chk_err("lin out");
    if (dmp) dump_vec("d_out.bin", bp(b_out), S * H);
    qk::add_inplace(bp(b_x), bp(b_out), S * H);
    chk_err("lin add");
    if (dmp) dump_vec("d_x_attn.bin", bp(b_x), S * H);
}

void QwenModel::run_full(LayerW& L, int li, int S, int kv_base)
{
    FullAttnW& fa = *L.fa;
    const int H = cfg.hidden_size;
    const int Hq = cfg.num_attention_heads;
    const int Hkv = cfg.num_key_value_heads;
    const int D = cfg.head_dim;
    const int kv_idx = (int)(std::lower_bound(full_layer_idx.begin(), full_layer_idx.end(), li) -
                             full_layer_idx.begin());

    cuda::matmul(b_h, fa.q_t, b_qq);      // [S, 2*Hq*D] = 每 token [q0 g0 q1 g1 ...] 按头交错
    cuda::matmul(b_h, fa.k_t, b_kv);
    cuda::matmul(b_h, fa.v_t, b_kv2);
    // 按头拆分为连续 q 与 gate (HF: 每 token 内 [q0 g0 q1 g1 ...] 交错)
    qk::split_qgate(bp(b_qq), bp(b_q), bp(b_gate), S, Hq, D);
    chk_err("full split_qgate");

    qk::rms_norm_1pw(bp(b_q), bp(fa.q_norm), bp(b_q), S * Hq, D, cfg.rms_norm_eps);
    qk::rms_norm_1pw(bp(b_kv), bp(fa.k_norm), bp(b_kv), S * Hkv, D, cfg.rms_norm_eps);
    qk::rope_text(bp(b_q), bp(b_q), d_pos_t.device_ptr(), d_pos_h.device_ptr(),
                  d_pos_w.device_ptr(), d_inv_freq.device_ptr(), d_section.device_ptr(),
                  S, Hq, D, cfg.rot_dim);
    qk::rope_text(bp(b_kv), bp(b_kv), d_pos_t.device_ptr(), d_pos_h.device_ptr(),
                  d_pos_w.device_ptr(), d_inv_freq.device_ptr(), d_section.device_ptr(),
                  S, Hkv, D, cfg.rot_dim);

    KVState& st = kv[kv_idx];
    const size_t row_elems = (size_t)Hkv * D;
    cudaMemcpyAsync(bp(st.k) + (size_t)kv_base * row_elems, bp(b_kv),
                    (size_t)S * row_elems * sizeof(qk::bf16), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(bp(st.v) + (size_t)kv_base * row_elems, bp(b_kv2),
                    (size_t)S * row_elems * sizeof(qk::bf16), cudaMemcpyDeviceToDevice);

    qk::attn_forward(bp(b_q), bp(st.k), bp(st.v), bp(b_attn), S, kv_base + S,
                     Hq, Hkv, D, kv_base, true, 1.0f / sqrtf((float)D));

    qk::sigmoid_mul(bp(b_attn), bp(b_gate), bp(b_attn), S * Hq * D);
    cuda::matmul(b_attn, fa.o_t, b_out);
    qk::add_inplace(bp(b_x), bp(b_out), S * H);

    st.len = kv_base + S;
}

void QwenModel::run_mlp(LayerW& L, int li, int S)
{
    const int H = cfg.hidden_size;
    const int I = cfg.intermediate_size;
    const bool dmp = g_dump && li == 0;
    cuda::matmul(b_h, L.mlp.gate_t, b_mlp);
    cuda::matmul(b_h, L.mlp.up_t, b_up);
    qk::silu(bp(b_mlp), bp(b_mlp), S * I);
    if (dmp) dump_vec("d_mlp_silu.bin", bp(b_mlp), S * I);
    qk::mul_inplace(bp(b_mlp), bp(b_up), S * I);
    if (dmp) dump_vec("d_mlp_mul.bin", bp(b_mlp), S * I);
    cuda::matmul(b_mlp, L.mlp.down_t, b_out);
    qk::add_inplace(bp(b_x), bp(b_out), S * H);
    if (dmp) dump_vec("d_x_layer0.bin", bp(b_x), S * H);
}
