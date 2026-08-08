// ============================================================
// Qwen3.5-0.8B 聊天 CLI (TensorN + CUDA, BF16 推理)
//
// 用法:
//   qwen35_chat [--model <目录>] [--tokenizer <tokenizer.json>]
//               [--image <图片路径>] [--max-new <n>]
// ============================================================

#include "qwen_model.hpp"
#include "vision.hpp"
#include "image.hpp"
#include "tokenizer.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using TensorN::CudaTensor;

namespace
{
    std::string trim(const std::string& s)
    {
        size_t a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) return "";
        size_t b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    }

    // 按 UTF-8 原始字节读取一行 (控制台 CP65001 或文件重定向均可用)
    std::string read_line_utf8()
    {
        std::string line;
        char c;
        while (std::cin.get(c))
        {
            if (c == '\n') break;
            line += c;
        }
        return line;
    }

    bool starts_with(const std::string& s, const std::string& p)
    {
        return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
    }

    std::string replace_all(std::string s, const std::string& from, const std::string& to)
    {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos)
        {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
        return s;
    }

    const std::string SYSTEM_PROMPT =
        "你是通义千问（Qwen），一个由阿里云开发的乐于助人、诚实无害的AI助手。"
        "请用简洁、准确的语言回答用户的问题。";

    // 生成 3*S 的 mrope 位置 (T/H/W) 与 rope_delta
    void build_pos3(const std::vector<int>& ids, int img_tok_id,
                    int grid_h, int grid_w, int merge, std::vector<int>& pos3, int& delta)
    {
        int S = (int)ids.size();
        pos3.assign(3 * S, 0);
        std::vector<std::pair<int, int>> img_ranges;
        int i = 0;
        while (i < S)
        {
            if (ids[i] == img_tok_id)
            {
                int j = i;
                while (j < S && ids[j] == img_tok_id) j++;
                img_ranges.push_back({i, j});
                i = j;
            }
            else i++;
        }
        std::vector<std::pair<int, int>> segs;
        int prev = 0;
        for (auto& r : img_ranges)
        {
            if (prev < r.first) segs.push_back({prev, r.first});
            segs.push_back(r);
            prev = r.second;
        }
        if (prev < S) segs.push_back({prev, S});

        int cur = 0;
        for (auto& sg : segs)
        {
            int a = sg.first, b = sg.second;
            if (a == b) continue;
            bool img = ids[a] == img_tok_id;
            if (!img)
            {
                for (int t = a; t < b; t++)
                {
                    pos3[t] = cur + (t - a);
                    pos3[S + t] = cur + (t - a);
                    pos3[2 * S + t] = cur + (t - a);
                }
                cur += b - a;
            }
            else
            {
                int llm_h = grid_h / merge, llm_w = grid_w / merge;
                int n = llm_h * llm_w;
                for (int k = 0; k < n; k++)
                {
                    int ph = k / llm_w, pw = k % llm_w;
                    pos3[a + k] = cur;
                    pos3[S + a + k] = cur + ph;
                    pos3[2 * S + a + k] = cur + pw;
                }
                cur += std::max(grid_h, grid_w) / merge;
            }
        }
        long long maxpos = 0;
        for (int t = 0; t < 3 * S; t++)
            if (pos3[t] > maxpos) maxpos = pos3[t];
        delta = (int)(maxpos + 1 - S);
    }
}

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif

    std::string model_dir = "Qwen3.5-0.8B";
    std::string tokenizer_path = "tokenizer/tokenizer.json";
    std::string image_path;
    int max_new = 512;
    bool selftest = false;
    bool timing = false;

    for (int i = 1; i < argc; i++)
    {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--tokenizer") && i + 1 < argc) tokenizer_path = argv[++i];
        else if (!std::strcmp(argv[i], "--image") && i + 1 < argc) image_path = argv[++i];
        else if (!std::strcmp(argv[i], "--max-new") && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--selftest")) selftest = true;
        else if (!std::strcmp(argv[i], "--timing")) timing = true;
        else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h"))
        {
            std::cout << "用法: qwen35_chat [--model <目录>] [--tokenizer <json>] "
                         "[--image <图片>] [--max-new <n>] [--selftest] [--timing]\n";
            return 0;
        }
    }

    std::string safetensors = model_dir + "/model.safetensors-00001-of-00001.safetensors";

    try
    {
        // ---- 加载 ----
        std::cout << "加载 tokenizer... " << std::endl;
        BPETokenizer tokenizer;
        if (!tokenizer.load(tokenizer_path))
        {
            std::cerr << "无法加载 " << tokenizer_path << std::endl;
            return 1;
        }

        std::cout << "加载模型权重 (BF16)... " << std::endl;
        auto tl0 = std::chrono::high_resolution_clock::now();
        QwenModel model;
        if (!model.load(model_dir, safetensors))
        {
            std::cerr << "模型加载失败" << std::endl;
            return 1;
        }
        VisionModel vision;
        if (!vision.load(model_dir, safetensors))
        {
            std::cerr << "视觉编码器加载失败" << std::endl;
            return 1;
        }
        auto tl1 = std::chrono::high_resolution_clock::now();
        std::cout << "模型加载完成: "
                  << (int)std::chrono::duration<double>(tl1 - tl0).count() << " s" << std::endl;

        std::cout << "模型就绪: " << model.cfg.num_hidden_layers << " 层, "
                  << model.cfg.hidden_size << " 隐藏维, vocab " << model.cfg.vocab_size
                  << ", 线性层 " << model.n_linear
                  << ", full 层 " << model.full_layer_idx.size() << std::endl;
        model.debug_timing = timing;

        // ---- 单 token 自测 ----
        if (selftest)
        {
            model.debug_timing = timing;
            int tid = tokenizer.token_id("hello");   // ASCII, 避免控制台编码问题
            std::cout << "[selftest] 单 token 前向 (" << tid << ")..." << std::endl;
            std::vector<int> ids = {tid};
            std::vector<int> pos3 = {0, 0, 0};
            auto t0 = std::chrono::high_resolution_clock::now();
            int tok = model.forward(ids, nullptr, -1, pos3);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "[selftest] 单 token: " << ms << " ms, 输出 token " << tok
                      << " (\"" << tokenizer.decode_one(tok) << "\")" << std::endl;

            // 再测一个小 prefill (10 tokens)
            std::vector<int> ids2 = {tid, tid, tid, tid, tid, tid, tid, tid, tid, tid};
            std::vector<int> pos3b(3 * 10);
            // pos3 布局: [T(10), H(10), W(10)], 文本三通道位置相同
            for (int i = 0; i < 10; i++)
            {
                pos3b[i] = 1 + i;
                pos3b[10 + i] = 1 + i;
                pos3b[20 + i] = 1 + i;
            }
            model.reset_state();
            t0 = std::chrono::high_resolution_clock::now();
            tok = model.forward(ids2, nullptr, -1, pos3b);
            t1 = std::chrono::high_resolution_clock::now();
            ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "[selftest] 10-token prefill: " << ms << " ms, 输出 token " << tok
                      << std::endl;
            return 0;
        }

        const int eos1 = 248044;  // <|endoftext|>
        const int eos2 = tokenizer.token_id("<|im_end|>");
        const int img_tok = model.cfg.image_token_id;

        std::vector<std::pair<std::string, std::string>> history;  // (user, assistant)

        // ---- 首轮图片 ----
        VisionPrep prep;
        bool has_img = false;
        if (!image_path.empty())
        {
            LoadedImage img;
            if (!load_image(image_path, img))
            {
                std::cerr << "无法加载图片: " << image_path << std::endl;
                return 1;
            }
            prep = prepare_image(img);
            std::cout << "图片已加载: " << img.w << "x" << img.h << " -> 网格 "
                      << prep.grid_h << "x" << prep.grid_w << ", 视觉 token "
                      << prep.n_tokens << std::endl;
            has_img = true;
        }

        // 多轮 KV 缓存: 首轮(或带图/重置后)全量 prefill;
        // 后续轮次保留 KV cache 与线性层状态, 只 prefill 新增片段。
        bool first_generation = true;

        auto generate = [&](const std::string& user_text, bool with_img) -> std::string
        {
            const bool full_prefill = first_generation || with_img;
            std::string content = user_text;
            if (with_img)
            {
                std::string pad;
                for (int i = 0; i < prep.n_tokens; i++) pad += "<|image_pad|>";
                content = "<|vision_start|>" + pad + "<|vision_end|>\n" + user_text;
            }

            std::string new_text;
            if (full_prefill)
            {
                new_text = "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n";
                for (auto& h : history)
                {
                    new_text += "<|im_start|>user\n" + h.first + "<|im_end|>\n";
                    new_text += "<|im_start|>assistant\n" + h.second + "<|im_end|>\n";
                }
                new_text += "<|im_start|>user\n" + content + "<|im_end|>\n";
                new_text += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
            }
            else
            {
                // 增量: 上一轮回复 <|im_end|> 之后的内容 (回复本体已在缓存中)
                new_text = "<|im_end|>\n<|im_start|>user\n" + content + "<|im_end|>\n" +
                           "<|im_start|>assistant\n<think>\n\n</think>\n\n";
            }

            auto ids = tokenizer.encode(new_text, false);
            const int S = (int)ids.size();
            if (model.debug_timing)
            {
                std::cout << "[debug prompt] " << new_text << std::endl;
                std::cout << "[debug ids " << S << "]: ";
                for (int t = 0; t < S && t < 80; t++) std::cout << ids[t] << " ";
                std::cout << std::endl;
            }

            // 图片特征
            CudaTensor<TensorN::bfloat16>* feat = nullptr;
            int img_at = -1;
            if (with_img)
            {
                int first = -1, cnt = 0;
                for (int t = 0; t < S; t++)
                {
                    if (ids[t] == img_tok) { if (first < 0) first = t; cnt++; }
                }
                if (cnt != prep.n_tokens)
                {
                    std::cerr << "image_pad 数量不匹配: " << cnt << " vs " << prep.n_tokens
                              << std::endl;
                    return "";
                }
                img_at = first;
            }

            std::vector<int> pos3;
            int delta = 0;
            if (full_prefill)
            {
                build_pos3(ids, img_tok, prep.grid_h, prep.grid_w, vision.spatial_merge,
                           pos3, delta);
            }
            else
            {
                // 文本增量: 位置 = past_len + i + rope_delta (继续前序上下文)
                delta = model.rope_delta;
                int base = model.past_len;
                pos3.resize(3 * S);
                for (int t = 0; t < S; t++)
                {
                    pos3[t] = base + t + delta;
                    pos3[S + t] = base + t + delta;
                    pos3[2 * S + t] = base + t + delta;
                }
            }

            // ---- 计算图片特征 ----
            static CudaTensor<TensorN::bfloat16> img_feat;
            if (with_img)
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                img_feat = vision.forward(prep);
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                std::cout << "[视觉编码 " << (int)ms << " ms]" << std::endl;
                feat = &img_feat;
            }

            // ---- 生成 ----
            if (full_prefill)
            {
                model.reset_state();
                model.rope_delta = delta;   // 供后续增量轮次使用
            }
            std::string out;
            auto t0 = std::chrono::high_resolution_clock::now();
            int tok = model.forward(ids, feat, img_at, pos3);
            auto tp = std::chrono::high_resolution_clock::now();
            double pms = std::chrono::duration<double, std::milli>(tp - t0).count();
            if (S > 1)
                std::cout << "[预填充 " << S << " tokens, " << (int)pms << " ms]"
                          << std::endl;
            int n = 0;
            while (n < max_new && tok != eos1 && tok != eos2)
            {
                std::string cur = tokenizer.decode_one(tok);
                if (!cur.empty()) { std::cout << cur << std::flush; out += cur; }
                n++;
                std::vector<int> one = {tok};
                std::vector<int> p3 = {model.past_len + delta,
                                       model.past_len + delta,
                                       model.past_len + delta};
                tok = model.forward(one, nullptr, -1, p3);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            std::cout << std::endl << "[生成 " << n << " tokens, " << sec << " s, "
                      << n / std::max(sec, 1e-6) << " tok/s]" << std::endl;

            // 清理输出
            out = replace_all(out, "<|im_end|>", "");
            out = replace_all(out, "<|endoftext|>", "");
            first_generation = false;
            return trim(out);
        };

        // ---- 对话循环 ----
        std::cout << "\n对话开始 (输入 /quit 退出, /image <路径> 附图片, /reset 清空历史)\n";
        std::string pending_image;
        while (true)
        {
            std::cout << "\n>>> " << std::flush;
            std::string line = trim(read_line_utf8());
            if (line.empty() && !std::cin) break;
            if (line.empty()) continue;
            if (line == "/quit" || line == "/exit") break;
            if (line == "/reset") { history.clear(); pending_image.clear(); first_generation = true; continue; }
            if (starts_with(line, "/image "))
            {
                pending_image = trim(line.substr(7));
                std::cout << "[图片已附加: " << pending_image << ", 请输入问题]" << std::endl;
                continue;
            }
            bool with_img = has_img || !pending_image.empty();
            if (!pending_image.empty())
            {
                LoadedImage img;
                if (!load_image(pending_image, img))
                    std::cerr << "无法加载图片: " << pending_image << std::endl;
                else
                {
                    prep = prepare_image(img);
                    std::cout << "图片 " << img.w << "x" << img.h << " -> 网格 "
                              << prep.grid_h << "x" << prep.grid_w << ", 视觉 token "
                              << prep.n_tokens << std::endl;
                    has_img = true;
                }
                pending_image.clear();
            }
            std::cout << "\n[用户] " << line << std::endl;
            std::cout << "[Qwen] ";
            std::string reply = generate(line, with_img);
            if (!reply.empty())
                history.emplace_back(line, reply);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "\n错误: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
