// ============================================================
// 通用 Dense Transformer (Llama 风格) 实现
// BF16 权重 + FP32 累加, 复用 qk:: CUDA 内核
// ============================================================

#include "dense_model.hpp"
#include "fast_ops.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

using namespace TensorN;

typedef TensorN::bfloat16 B16;

namespace
{
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

    inline qk::bf16* bp(CudaTensor<B16>& t) { return reinterpret_cast<qk::bf16*>(t.device_ptr()); }
    inline const qk::bf16* bp(const CudaTensor<B16>& t)
    {
        return reinterpret_cast<const qk::bf16*>(t.device_ptr());
    }

    const SafeTensor* find_st(const std::unordered_map<std::string, SafeTensor>& raw,
                              const std::vector<std::string>& names)
    {
        for (auto& n : names)
        {
            auto it = raw.find(n);
            if (it != raw.end()) return &it->second;
        }
        return nullptr;
    }

    const SafeTensor& need_st(const std::unordered_map<std::string, SafeTensor>& raw,
                              const std::vector<std::string>& names)
    {
        const SafeTensor* st = find_st(raw, names);
        if (!st)
        {
            std::string msg = "missing weight, tried:";
            for (auto& n : names) msg += "\n  " + n;
            throw std::runtime_error(msg);
        }
        return *st;
    }

    std::vector<qk::bf16> to_host_bf16(const SafeTensor& st)
    {
        if (st.dtype != "BF16") throw std::runtime_error("expected BF16, got " + st.dtype);
        size_t n = st.data.size() / 2;
        std::vector<qk::bf16> out(n);
        std::memcpy(out.data(), st.data.data(), st.data.size());
        return out;
    }

    CudaTensor<B16> mat_plain(const std::unordered_map<std::string, SafeTensor>& raw,
                              const std::vector<std::string>& names)
    {
        const SafeTensor& st = need_st(raw, names);
        auto v = to_host_bf16(st);
        std::vector<size_t> sh(st.shape.begin(), st.shape.end());
        CudaTensor<B16> t(sh);
        t.copyFromHost(reinterpret_cast<const B16*>(v.data()), v.size());
        return t;
    }

    // 把多个 [rows, in] 权重按行拼成一个 [total_rows, in] 张量
    // (qkv / gate+up 合并, 一次 GEMM 完成多路投影)
    CudaTensor<B16> pack_rows(const std::unordered_map<std::string, SafeTensor>& raw,
                              const std::vector<std::string>& names,
                              int total_rows, int in_dim)
    {
        std::vector<qk::bf16> buf((size_t)total_rows * in_dim);
        size_t off = 0;
        for (auto& n : names)
        {
            const SafeTensor& st = need_st(raw, {n});
            if (st.dtype != "BF16")
                throw std::runtime_error("pack_rows: " + n + " is not BF16");
            if (st.shape.size() != 2 || (int)st.shape[1] != in_dim)
                throw std::runtime_error("pack_rows: shape mismatch for " + n);
            size_t rows = (size_t)st.shape[0];
            if (off + rows > (size_t)total_rows)
                throw std::runtime_error("pack_rows: too many rows for " + n);
            std::memcpy(buf.data() + off * in_dim, st.data.data(),
                        rows * (size_t)in_dim * sizeof(qk::bf16));
            off += rows;
        }
        if (off != (size_t)total_rows)
            throw std::runtime_error("pack_rows: row count mismatch");
        CudaTensor<B16> t({(size_t)total_rows, (size_t)in_dim});
        t.copyFromHost(reinterpret_cast<const B16*>(buf.data()), buf.size());
        return t;
    }

    CudaTensor<B16> vec1(const std::unordered_map<std::string, SafeTensor>& raw,
                         const std::vector<std::string>& names)
    {
        const SafeTensor& st = need_st(raw, names);
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
} // namespace

bool DenseConfig::load(const std::string& config_path)
{
    std::ifstream f(config_path);
    if (!f.is_open()) return false;
    nlohmann::json j;
    f >> j;

    hidden_size = j.at("hidden_size").get<int>();
    intermediate_size = j.at("intermediate_size").get<int>();
    num_hidden_layers = j.at("num_hidden_layers").get<int>();
    vocab_size = j.at("vocab_size").get<int>();
    rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
    rope_theta = j.value("rope_theta", 10000.0f);
    max_position_embeddings = j.value("max_position_embeddings", 32768);
    tie_word_embeddings = j.value("tie_word_embeddings", false);

    if (j.contains("attention"))
    {
        auto& a = j["attention"];
        num_attention_heads = a.value("num_heads", 0);
        num_key_value_heads = a.value("num_kv_heads", num_attention_heads);
        if (a.contains("gqa"))
        {
            head_dim = a["gqa"].value("head_dim", 0);
            rot_dim = a["gqa"].value("rotary_dim", head_dim);
            // 该配置格式 (attention.gqa) 的模型使用 GPT-J 交错式 RoPE;
            // 标准 Llama config (无 attention 块) 仍默认半维配对
            rope_interleaved = a["gqa"].value("rope_interleaved", true);
        }
        if (head_dim == 0)
            head_dim = hidden_size / std::max(1, num_attention_heads);
        if (rot_dim == 0) rot_dim = head_dim;
    }
    else
    {
        num_attention_heads = j.value("num_attention_heads", 0);
        num_key_value_heads = j.value("num_key_value_heads", num_attention_heads);
        head_dim = j.value("head_dim", hidden_size / std::max(1, num_attention_heads));
        rot_dim = j.value("rotary_dim", head_dim);
        rope_interleaved = j.value("rope_interleaved", false);
    }
    norm_1pw = j.value("norm_1pw", j.value("rms_norm_1pw", false));

    if (num_attention_heads <= 0 || head_dim <= 0 || num_key_value_heads <= 0)
        throw std::runtime_error("config: invalid attention parameters");
    if (num_attention_heads % num_key_value_heads != 0)
        throw std::runtime_error("config: num_attention_heads must be divisible by num_kv_heads");
    if (rot_dim <= 0 || rot_dim > head_dim || (rot_dim & 1))
        throw std::runtime_error("config: invalid rotary_dim");
    return true;
}

bool DenseModel::load(const std::string& model_dir, const std::string& safetensors_path)
{
    if (!cfg.load(model_dir + "/config.json"))
        return false;

    {
        size_t f = 0, t = 0;
        cudaMemGetInfo(&f, &t);
        std::cout << "GPU 显存: 空闲 " << f / (1024 * 1024) << " MB / 共 "
                  << t / (1024 * 1024) << " MB" << std::endl;
    }

    auto raw = load_safetensors_raw(safetensors_path);
    std::cout << "  权重文件读取完成 (" << raw.size() << " 个张量)" << std::endl;

    const std::string L = "model.layers.";
    const std::string A = ".attention.";
    const std::string M = ".mlp.";

    embed = mat_plain(raw, {"model.embed_tokens.weight", "transformer.wte.weight"});

    bool has_head = raw.find("lm_head.weight") != raw.end();
    if (has_head)
    {
        lm_head_w = mat_plain(raw, {"lm_head.weight"});
        int V = (int)lm_head_w.shape()[0], H = (int)lm_head_w.shape()[1];
        lm_head_q.q = CudaTensor<int8_t>({(size_t)V, (size_t)H});
        lm_head_q.scale = CudaTensor<float>({(size_t)V});
        fo::quantize_w8a16(bp(lm_head_w), lm_head_q.q.device_ptr(),
                           lm_head_q.scale.device_ptr(), V, H);
    }
    else
        tied_lm_head = true;

    final_norm = vec1(raw, {"model.norm.weight", "transformer.ln_f.weight"});

    {
        auto es = embed.shape();
        if (es.size() != 2 || (int)es[0] != cfg.vocab_size || (int)es[1] != cfg.hidden_size)
            throw std::runtime_error("embed_tokens shape mismatch");
        if (final_norm.size() != (size_t)cfg.hidden_size)
            throw std::runtime_error("final norm shape mismatch");
        if (has_head)
        {
            auto hs = lm_head_w.shape();
            if (hs.size() != 2 || (int)hs[0] != cfg.vocab_size ||
                (int)hs[1] != cfg.hidden_size)
                throw std::runtime_error("lm_head shape mismatch");
        }
    }

    layers.resize(cfg.num_hidden_layers);
    for (int i = 0; i < cfg.num_hidden_layers; i++)
    {
        if (i % 4 == 0)
            std::cout << "  加载层 " << i << "/" << cfg.num_hidden_layers << std::endl;
        DenseLayerW& Lw = layers[i];
        std::string p = L + std::to_string(i);
        Lw.in_norm = vec1(raw, {p + ".input_norm.weight", p + ".input_layernorm.weight"});
        Lw.post_norm = vec1(raw, {p + ".post_attention_norm.weight",
                                  p + ".post_attention_layernorm.weight"});

        const int H = cfg.hidden_size, I = cfg.intermediate_size;
        const int Q = cfg.num_attention_heads * cfg.head_dim;
        const int KV = cfg.num_key_value_heads * cfg.head_dim;

        // 保持 [out, in] 原始布局; qkv / gate+up 按行合并, 减少 GEMM 次数
        Lw.attn.qkv_w = pack_rows(raw, {p + A + "q_proj.weight",
                                        p + A + "k_proj.weight",
                                        p + A + "v_proj.weight"},
                                  Q + 2 * KV, H);
        Lw.attn.o_w = mat_plain(raw, {p + A + "o_proj.weight",
                                      p + ".self_attn.o_proj.weight"});

        Lw.mlp.gu_w = pack_rows(raw, {p + M + "gate.weight", p + M + "up.weight"},
                                2 * I, H);
        Lw.mlp.down_w = mat_plain(raw, {p + M + "down.weight",
                                        p + M + "down_proj.weight"});

        // 形状检查 (原始 [out, in] 布局)
        auto chk2 = [&](const CudaTensor<B16>& t, int out, int in, const char* name) {
            auto& sh = t.shape();
            if (sh.size() != 2 || (int)sh[0] != out || (int)sh[1] != in)
                throw std::runtime_error(std::string("weight shape mismatch: ") + name +
                                         " (layer " + std::to_string(i) + ")");
        };
        chk2(Lw.attn.qkv_w, Q + 2 * KV, H, "qkv_proj");
        chk2(Lw.attn.o_w, H, Q, "o_proj");
        chk2(Lw.mlp.gu_w, 2 * I, H, "mlp.gu");
        chk2(Lw.mlp.down_w, H, I, "mlp.down");

        // W8A16: decode 用 int8 权重, prefill 继续用 BF16 张量核
        auto quantize = [](const CudaTensor<B16>& w, QuantW& qw) {
            int N = (int)w.shape()[0], K = (int)w.shape()[1];
            qw.q = CudaTensor<int8_t>({(size_t)N, (size_t)K});
            qw.scale = CudaTensor<float>({(size_t)N});
            fo::quantize_w8a16(bp(w), qw.q.device_ptr(), qw.scale.device_ptr(),
                               N, K);
        };
        quantize(Lw.attn.qkv_w, Lw.attn.qkv_q);
        quantize(Lw.attn.o_w, Lw.attn.o_q);
        quantize(Lw.mlp.gu_w, Lw.mlp.gu_q);
        quantize(Lw.mlp.down_w, Lw.mlp.down_q);
    }

    // RoPE 逆频率 [rot_dim/2]
    const int half = cfg.rot_dim / 2;
    std::vector<float> inv_freq(half);
    for (int i = 0; i < half; i++)
        inv_freq[i] = 1.0f / powf(cfg.rope_theta, (float)i / (float)half);
    d_inv_freq = CudaTensor<float>({(size_t)half});
    d_inv_freq.copyFromHost(inv_freq.data(), inv_freq.size());

    // KV cache
    if (max_ctx <= 0) max_ctx = 8192;
    max_ctx = std::min(max_ctx, cfg.max_position_embeddings);
    const size_t row = (size_t)cfg.num_key_value_heads * cfg.head_dim;
    kv.resize(cfg.num_hidden_layers);
    for (auto& s : kv)
    {
        s.k = CudaTensor<B16>({(size_t)max_ctx, row});
        s.v = CudaTensor<B16>({(size_t)max_ctx, row});
    }

    // 输入索引/位置缓冲
    d_ids = CudaTensor<int>({(size_t)max_ctx});
    d_pos = CudaTensor<int>({(size_t)max_ctx});

    std::cout << "  权重上传完成" << std::endl;
    past_len = 0;
    return true;
}

void DenseModel::reset_state()
{
    past_len = 0;
}

void DenseModel::ensure_buffers(int S)
{
    if (buf_s >= S) return;
    buf_s = S;
    const size_t H = (size_t)cfg.hidden_size;
    const size_t I = (size_t)cfg.intermediate_size;
    const size_t Q = (size_t)cfg.num_attention_heads * cfg.head_dim;
    const size_t KV = (size_t)cfg.num_key_value_heads * cfg.head_dim;
    const size_t Sm = (size_t)S;

    b_x = CudaTensor<B16>({Sm, H});
    b_h = CudaTensor<B16>({Sm, H});
    b_q = CudaTensor<B16>({Sm, Q});
    b_k = CudaTensor<B16>({Sm, KV});
    b_v = CudaTensor<B16>({Sm, KV});
    b_attn = CudaTensor<B16>({Sm, Q});
    b_out = CudaTensor<B16>({Sm, H});
    b_gate = CudaTensor<B16>({Sm, I});
    b_up = CudaTensor<B16>({Sm, I});
    b_qkv = CudaTensor<B16>({Sm, Q + 2 * KV});
    b_gu = CudaTensor<B16>({Sm, 2 * I});
    b_hid = CudaTensor<B16>({Sm, I});
    b_logits = CudaTensor<B16>({(size_t)cfg.vocab_size});
}

int DenseModel::forward(const std::vector<int>& ids)
{
    const int S = (int)ids.size();
    if (S <= 0) throw std::runtime_error("empty input");
    if (past_len + S > max_ctx)
        throw std::runtime_error("context too long (max " + std::to_string(max_ctx) + ")");

    const int H = cfg.hidden_size;
    const bool dbg = debug_timing && S > 1;

    ensure_buffers(S);

    std::vector<int> pos(S);
    for (int i = 0; i < S; i++) pos[i] = past_len + i;
    cudaMemcpyAsync(d_ids.device_ptr(), ids.data(), (size_t)S * sizeof(int),
                    cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_pos.device_ptr(), pos.data(), (size_t)S * sizeof(int),
                    cudaMemcpyHostToDevice);

    qk::embed_gather(bp(embed), d_ids.device_ptr(), bp(b_x), S, H);
    chk_err("embed");

    for (int li = 0; li < (int)layers.size(); li++)
    {
        StageTimer st(dbg, "layer");
        run_layer(li, S);
        chk_err("layer");
    }

    // 只归一化最后一行: 输出头只消费最后一个 token, 省掉 (S-1)*H 的无用写
    const qk::bf16* last_x = bp(b_x) + (size_t)(S - 1) * H;
    if (cfg.norm_1pw)
        qk::rms_norm_1pw(last_x, bp(final_norm), bp(b_h), 1, H, cfg.rms_norm_eps);
    else
        qk::rms_norm(last_x, bp(final_norm), bp(b_h), 1, H, cfg.rms_norm_eps);
    const auto& head = tied_lm_head ? embed : lm_head_w;
    if (fast_ops && use_w8a16 && lm_head_q.q.size() > 0)
        fo::gemv_w8a16(bp(b_h), lm_head_q.q.device_ptr(),
                       lm_head_q.scale.device_ptr(), bp(b_logits),
                       cfg.vocab_size, H);
    else if (fast_ops)
        fo::gemv_bf16(bp(b_h), bp(head), bp(b_logits), cfg.vocab_size, H);
    else
        qk::lm_head(bp(b_h), bp(head), bp(b_logits), cfg.vocab_size, H);
    int tok = qk::argmax(bp(b_logits), cfg.vocab_size);
    chk_err("lm_head");

    if (debug_logits)
    {
        std::vector<qk::bf16> lg(cfg.vocab_size);
        cudaMemcpy(lg.data(), bp(b_logits), (size_t)cfg.vocab_size * 2,
                   cudaMemcpyDeviceToHost);
        std::vector<std::pair<float, int>> top;
        top.reserve(cfg.vocab_size);
        for (int i = 0; i < cfg.vocab_size; i++)
            top.emplace_back(fp::bfloat16_bits_to_float(
                                 *reinterpret_cast<const uint16_t*>(&lg[i])), i);
        int k = std::min(10, (int)top.size());
        std::partial_sort(top.begin(), top.begin() + k, top.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        std::cout << "  [logits top" << k << "]: ";
        for (int i = 0; i < k; i++)
            std::cout << top[i].second << "(" << top[i].first << ") ";
        std::cout << std::endl;
    }

    past_len += S;
    return tok;
}

void DenseModel::run_layer(int li, int S)
{
    DenseLayerW& L = layers[li];
    const int H = cfg.hidden_size;
    const int Hq = cfg.num_attention_heads;
    const int Hkv = cfg.num_key_value_heads;
    const int D = cfg.head_dim;
    const int I = cfg.intermediate_size;
    const int Q = Hq * D;
    const int KV = Hkv * D;
    const int NQKV = Q + 2 * KV;
    const int base = past_len;
    const float scale = 1.0f / sqrtf((float)D);

    auto norm_into = [&](const TensorN::CudaTensor<B16>& w, TensorN::CudaTensor<B16>& dst) {
        if (cfg.norm_1pw)
            qk::rms_norm_1pw(bp(b_x), bp(w), bp(dst), S, H, cfg.rms_norm_eps);
        else
            qk::rms_norm(bp(b_x), bp(w), bp(dst), S, H, cfg.rms_norm_eps);
    };

    norm_into(L.in_norm, b_h);

    if (fast_ops)
    {
        // decode (S=1) 走 W8A16 GEMV; prefill 继续用 BF16 cuBLAS
        auto linear = [&](const qk::bf16* x, const QuantW& qw,
                          const qk::bf16* w, qk::bf16* y,
                          int N, int K, const qk::bf16* add, float beta) {
            if (S == 1 && use_w8a16)
                fo::gemv_w8a16(x, qw.q.device_ptr(), qw.scale.device_ptr(),
                               y, N, K, add);
            else
                fo::gemm_nt_bf16(x, w, y, S, N, K, beta);
        };

        // 一次 GEMM 产出 [Q | K | V]
        linear(bp(b_h), L.attn.qkv_q, bp(L.attn.qkv_w), bp(b_qkv),
               NQKV, H, nullptr, 0.f);
        // Q/K RoPE + K/V 直接写 cache, 不再经过独立 b_k/b_v
        fo::qkv_rope_kv(bp(b_qkv), d_pos.device_ptr(), d_inv_freq.device_ptr(),
                        bp(b_q), bp(kv[li].k), bp(kv[li].v), base, S,
                        Hq, Hkv, D, cfg.rot_dim, cfg.rope_interleaved ? 1 : 0);
        chk_err("qkv+rope");

        if (skip_attn)
            cudaMemsetAsync(bp(b_attn), 0, (size_t)S * Q * sizeof(qk::bf16));
        else if (S == 1 && fo::decode_attn_supported(D))
            fo::decode_attn(bp(b_q), bp(kv[li].k), bp(kv[li].v), bp(b_attn),
                            base + 1, Hq, Hkv, D, scale);
        else if (S > 1 && fo::flash_attn_supported(D))
            fo::flash_attn_prefill(bp(b_q), bp(kv[li].k), bp(kv[li].v), bp(b_attn),
                                   base, S, base + S, Hq, Hkv, D, scale);
        else
            qk::attn_forward(bp(b_q), bp(kv[li].k), bp(kv[li].v), bp(b_attn), S,
                             base + S, Hq, Hkv, D, base, true, scale);
        chk_err("attention");

        // o_proj: C = b_x, beta=1  =>  残差累加融进 GEMM
        linear(bp(b_attn), L.attn.o_q, bp(L.attn.o_w), bp(b_x),
               H, Q, bp(b_x), 1.f);

        norm_into(L.post_norm, b_h);

        // 一次 GEMM 产出 [gate | up], 再融合 SwiGLU
        linear(bp(b_h), L.mlp.gu_q, bp(L.mlp.gu_w), bp(b_gu),
               2 * I, H, nullptr, 0.f);
        fo::swiglu_bf16(bp(b_gu), bp(b_hid), S, I);

        // down_proj 同样把残差累加融进 GEMM
        linear(bp(b_hid), L.mlp.down_q, bp(L.mlp.down_w), bp(b_x),
               H, I, bp(b_x), 1.f);
    }
    else
    {
        // 参考路径: 独立 q/k/v + 独立 RoPE + 独立 KV 拷贝 + 独立残差
        fo::gemm_nt_bf16(bp(b_h), bp(L.attn.qkv_w), bp(b_q), S, Q, H, 0.f);
        fo::gemm_nt_bf16(bp(b_h), bp(L.attn.qkv_w) + (size_t)Q * H, bp(b_k),
                         S, KV, H, 0.f);
        fo::gemm_nt_bf16(bp(b_h), bp(L.attn.qkv_w) + (size_t)(Q + KV) * H,
                         bp(b_v), S, KV, H, 0.f);

        qk::rope_neox(bp(b_q), bp(b_q), d_pos.device_ptr(), d_inv_freq.device_ptr(),
                      S, Hq, D, cfg.rot_dim, cfg.rope_interleaved ? 1 : 0);
        qk::rope_neox(bp(b_k), bp(b_k), d_pos.device_ptr(), d_inv_freq.device_ptr(),
                      S, Hkv, D, cfg.rot_dim, cfg.rope_interleaved ? 1 : 0);

        DenseKVState& st = kv[li];
        cudaMemcpyAsync(bp(st.k) + (size_t)base * KV, bp(b_k),
                        (size_t)S * KV * sizeof(qk::bf16), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(bp(st.v) + (size_t)base * KV, bp(b_v),
                        (size_t)S * KV * sizeof(qk::bf16), cudaMemcpyDeviceToDevice);

        qk::attn_forward(bp(b_q), bp(st.k), bp(st.v), bp(b_attn), S, base + S,
                         Hq, Hkv, D, base, true, scale);

        fo::gemm_nt_bf16(bp(b_attn), bp(L.attn.o_w), bp(b_out), S, H, Q, 0.f);
        qk::add_inplace(bp(b_x), bp(b_out), S * H);

        norm_into(L.post_norm, b_h);

        fo::gemm_nt_bf16(bp(b_h), bp(L.mlp.gu_w), bp(b_gate), S, I, H, 0.f);
        fo::gemm_nt_bf16(bp(b_h), bp(L.mlp.gu_w) + (size_t)I * H, bp(b_up),
                         S, I, H, 0.f);
        qk::silu(bp(b_gate), bp(b_gate), S * I);
        qk::mul_inplace(bp(b_gate), bp(b_up), S * I);

        fo::gemm_nt_bf16(bp(b_gate), bp(L.mlp.down_w), bp(b_out), S, H, I, 0.f);
        qk::add_inplace(bp(b_x), bp(b_out), S * H);
    }
    chk_err("layer");
}
