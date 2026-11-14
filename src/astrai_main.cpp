// ============================================================
// AstrAI-V1-2B-instruct 聊天 CLI (Dense Transformer + CUDA BF16)
//
// 用法:
//   astrai_chat [--model <目录>] [--tokenizer <tokenizer.json>]
//               [--system <提示词>] [--max-new <n>] [--ctx <n>]
//               [--selftest] [--encode <文本>] [--timing]
// ============================================================

#include "dense_model.hpp"
#include "tokenizer.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <functional>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using TensorN::CudaTensor;

namespace
{
    const std::string IM_START = "<｜im▁start｜>";
    const std::string IM_END   = "<｜im▁end｜>";
    const std::string BOS_TOK  = "<｜begin▁of▁sentence｜>";
    const std::string EOS_TOK  = "<｜end▁of▁sentence｜>";

    std::string trim(const std::string& s)
    {
        size_t a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) return "";
        size_t b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    }

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

    std::string system_turn(const std::string& sys)
    {
        return IM_START + "system\n" + sys + IM_END + "\n";
    }
    std::string user_turn(const std::string& text)
    {
        return IM_START + "user\n" + text + IM_END + "\n";
    }
    std::string assistant_turn(const std::string& text)
    {
        return IM_START + "assistant\n" + text + IM_END + "\n";
    }
}

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif

    std::string model_dir = "AstrAI-V1-2B-instruct";
    std::string tokenizer_path;
    std::string sys_prompt = "You are AstrAI, a helpful, honest and harmless AI assistant.";
    std::string encode_text;
    std::string probe_text;
    std::string ids_text;
    int max_new = 512;
    int max_ctx = 8192;
    bool selftest = false;
    bool timing = false;
    bool do_encode = false;
    bool do_probe = false;
    bool do_ids = false;
    bool rope_interleaved = false;
    bool rope_neox = false;
    bool norm_1pw = false;
    bool full_prefill_always = false;
    bool use_ref = false;
    bool use_bf16 = false;
    bool skip_attn = false;
    int bench_n = 0;

    for (int i = 1; i < argc; i++)
    {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--tokenizer") && i + 1 < argc) tokenizer_path = argv[++i];
        else if (!std::strcmp(argv[i], "--system") && i + 1 < argc) sys_prompt = argv[++i];
        else if (!std::strcmp(argv[i], "--max-new") && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) max_ctx = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--encode") && i + 1 < argc)
        {
            encode_text = argv[++i];
            do_encode = true;
        }
        else if (!std::strcmp(argv[i], "--probe") && i + 1 < argc)
        {
            probe_text = argv[++i];
            do_probe = true;
        }
        else if (!std::strcmp(argv[i], "--ids") && i + 1 < argc)
        {
            ids_text = argv[++i];
            do_ids = true;
        }
        else if (!std::strcmp(argv[i], "--rope-interleaved")) rope_interleaved = true;
        else if (!std::strcmp(argv[i], "--rope-neox")) rope_neox = true;
        else if (!std::strcmp(argv[i], "--norm-1pw")) norm_1pw = true;
        else if (!std::strcmp(argv[i], "--full-prefill")) full_prefill_always = true;
        else if (!std::strcmp(argv[i], "--ref")) use_ref = true;
        else if (!std::strcmp(argv[i], "--bf16")) use_bf16 = true;
        else if (!std::strcmp(argv[i], "--skip-attn")) skip_attn = true;
        else if (!std::strcmp(argv[i], "--bench") && i + 1 < argc) bench_n = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--selftest")) selftest = true;
        else if (!std::strcmp(argv[i], "--timing")) timing = true;
        else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h"))
        {
            std::cout << "用法: astrai_chat [--model <目录>] [--tokenizer <json>]\n"
                         "                  [--system <提示词>] [--max-new <n>] [--ctx <n>]\n"
                         "                  [--selftest] [--encode <文本>] [--probe <文本>] [--timing]\n"
                         "                  [--rope-neox] [--rope-interleaved] [--norm-1pw]\n"
                         "                  [--full-prefill]  (每轮全量重算, 调试用)\n"
                         "                  [--ref]  (关闭快路径, 用参考实现对照)\n"
                         "                  [--bf16] (decode 也使用 BF16 权重, 关闭 W8A16)\n"
                         "                  [--bench <n>]  (测量 prefill/decode 吞吐)\n";
            return 0;
        }
    }
    if (tokenizer_path.empty()) tokenizer_path = model_dir + "/tokenizer.json";

    try
    {
        std::cout << "加载 tokenizer... " << std::endl;
        BPETokenizer tokenizer;
        if (!tokenizer.load(tokenizer_path))
        {
            std::cerr << "无法加载 " << tokenizer_path << std::endl;
            return 1;
        }

        if (do_encode)
        {
            auto ids = tokenizer.encode(encode_text, false);
            std::cout << "ids (" << ids.size() << "): ";
            for (int id : ids) std::cout << id << " ";
            std::cout << "\ndecode: " << tokenizer.decode(ids) << std::endl;
            return 0;
        }

        std::cout << "加载模型权重 (BF16)... " << std::endl;
        auto tl0 = std::chrono::high_resolution_clock::now();
        DenseModel model;
        model.max_ctx = max_ctx;
        model.debug_timing = timing;
        if (!model.load(model_dir, model_dir + "/model.safetensors"))
        {
            std::cerr << "模型加载失败" << std::endl;
            return 1;
        }
        if (rope_interleaved) model.cfg.rope_interleaved = true;
        if (rope_neox) model.cfg.rope_interleaved = false;
        if (norm_1pw) model.cfg.norm_1pw = true;
        model.fast_ops = !use_ref;
        model.use_w8a16 = !use_bf16;
        model.skip_attn = skip_attn;
        auto tl1 = std::chrono::high_resolution_clock::now();
        std::cout << "模型加载完成: "
                  << (int)std::chrono::duration<double>(tl1 - tl0).count() << " s" << std::endl;
        std::cout << "模型就绪: " << model.cfg.num_hidden_layers << " 层, "
                  << model.cfg.hidden_size << " 隐藏维, vocab " << model.cfg.vocab_size
                  << ", GQA " << model.cfg.num_attention_heads << "/"
                  << model.cfg.num_key_value_heads << ", head_dim "
                  << model.cfg.head_dim << ", ctx " << model.max_ctx
                  << ", backend "
                  << (model.fast_ops ? (model.use_w8a16 ? "fast/w8a16" : "fast/bf16")
                                     : "ref/bf16")
                  << std::endl;

        const int eos_im_end = tokenizer.token_id(IM_END);
        const int eos_sentence = tokenizer.token_id(EOS_TOK);

        if (selftest)
        {
            std::vector<int> ids = tokenizer.encode("Hello", false);
            std::cout << "[selftest] ids: ";
            for (int id : ids) std::cout << id << " ";
            std::cout << std::endl;
            model.debug_logits = true;
            auto t0 = std::chrono::high_resolution_clock::now();
            int tok = model.forward(ids);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "[selftest] " << ids.size() << " tokens: " << ms
                      << " ms, next token " << tok << " (\"" << tokenizer.decode_one(tok)
                      << "\")" << std::endl;
            return 0;
        }

        // 吞吐基准: 预热 -> prefill -> decode
        if (bench_n > 0)
        {
            std::string text = probe_text.empty()
                ? "The quick brown fox jumps over the lazy dog. "
                  "Artificial intelligence is transforming the way people work, learn, "
                  "and communicate. In a distant galaxy, a small spacecraft drifted "
                  "silently through the darkness while its crew slept."
                : probe_text;
            auto ids = tokenizer.encode(text, false);
            int hello = tokenizer.encode("Hello", false)[0];

            model.reset_state();
            model.forward({hello});
            model.reset_state();
            // 同一形状先跑一遍, 排除 cuBLAS 首次 shape 初始化 / 内核加载开销
            model.forward(ids);
            model.reset_state();

            auto t0 = std::chrono::high_resolution_clock::now();
            int tok = model.forward(ids);
            auto t1 = std::chrono::high_resolution_clock::now();
            double pre_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            for (int n = 1; n < bench_n; n++)
                tok = model.forward({tok});
            auto t2 = std::chrono::high_resolution_clock::now();
            double dec_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            int dec_n = bench_n > 1 ? bench_n - 1 : 0;

            std::cout << "[bench] backend=" << (model.fast_ops ? "fast" : "ref")
                      << " prompt=" << ids.size() << " tokens" << std::endl;
            std::cout << "[bench] prefill: " << pre_ms << " ms, "
                      << ids.size() / std::max(pre_ms / 1000.0, 1e-9) << " tok/s"
                      << std::endl;
            if (dec_n > 0)
                std::cout << "[bench] decode : " << dec_ms << " ms, "
                          << dec_n / std::max(dec_ms / 1000.0, 1e-9) << " tok/s"
                          << " (" << dec_ms / dec_n << " ms/token)" << std::endl;
            return 0;
        }

        // 直接输入 token ids, 打印下一步 logits top10 (与参考实现对比)
        if (do_ids)
        {
            std::vector<int> ids;
            std::string cur;
            for (char c : ids_text)
            {
                if (c == ',' || c == ' ' || c == '\t')
                {
                    if (!cur.empty()) { ids.push_back(std::atoi(cur.c_str())); cur.clear(); }
                }
                else cur += c;
            }
            if (!cur.empty()) ids.push_back(std::atoi(cur.c_str()));
            std::cout << "[ids] " << ids.size() << " tokens" << std::endl;
            model.debug_logits = true;
            int tok = model.forward(ids);
            std::cout << "[ids] greedy next: " << tok << " (\"" << tokenizer.decode_one(tok)
                      << "\")" << std::endl;
            return 0;
        }

        // 原始文本续写 (不走 chat 模板, 方便与参考实现逐 token 对比)
        if (do_probe)
        {
            auto ids = tokenizer.encode(probe_text, false);
            std::cout << "[probe] prompt ids (" << ids.size() << "): ";
            for (int id : ids) std::cout << id << " ";
            std::cout << std::endl;
            model.debug_logits = true;
            int tok = model.forward(ids);
            model.debug_logits = false;
            std::vector<int> gen = {tok};
            std::cout << "[probe] gen ids: " << tok;
            for (int n = 1; n < max_new; n++)
            {
                if (tok == eos_im_end || tok == eos_sentence) break;
                tok = model.forward({tok});
                gen.push_back(tok);
                std::cout << " " << tok;
            }
            std::cout << "\n[probe] text: " << tokenizer.decode(gen) << std::endl;
            return 0;
        }

        std::vector<std::pair<std::string, std::string>> history;
        bool first_generation = true;

        std::function<std::string(const std::string&)> run_generation;
        run_generation = [&](const std::string& user_text) -> std::string
        {
            std::string new_text;
            const bool full_prefill = first_generation || full_prefill_always;
            if (full_prefill)
            {
                model.reset_state();
                new_text = system_turn(sys_prompt);
                for (auto& h : history)
                {
                    new_text += user_turn(h.first);
                    new_text += assistant_turn(h.second);
                }
                new_text += user_turn(user_text);
            }
            else
            {
                new_text = IM_END + "\n" + user_turn(user_text);
            }
            new_text += IM_START + "assistant\n";

            auto ids = tokenizer.encode(new_text, false);
            const int S = (int)ids.size();
            if (S <= 0)
            {
                std::cerr << "空提示词" << std::endl;
                return "";
            }
            if (model.past_len + S > model.max_ctx)
            {
                if (first_generation)
                {
                    std::cerr << "提示词过长 (" << S << " > " << model.max_ctx
                              << "), 请增大 --ctx 或缩短输入" << std::endl;
                    return "";
                }
                std::cerr << "上下文超限 (" << model.past_len << " + " << S << " > "
                          << model.max_ctx << "), 自动重置历史" << std::endl;
                history.clear();
                first_generation = true;
                return run_generation(user_text);
            }

            if (model.debug_timing)
            {
                std::cout << "[debug prompt] " << new_text << std::endl;
                std::cout << "[debug ids " << S << "]: ";
                for (int t = 0; t < S && t < 80; t++) std::cout << ids[t] << " ";
                std::cout << std::endl;
            }

            std::cout << "[预填充 " << S << " tokens]" << std::endl;
            auto t0 = std::chrono::high_resolution_clock::now();
            int tok = model.forward(ids);
            auto tp = std::chrono::high_resolution_clock::now();
            double pms = std::chrono::duration<double, std::milli>(tp - t0).count();
            if (model.debug_timing)
                std::cout << "[prefill " << (int)pms << " ms]" << std::endl;

            std::vector<int> generated;
            size_t printed = 0;
            int n = 0;
            while (n < max_new && tok != eos_im_end && tok != eos_sentence && tok >= 0)
            {
                generated.push_back(tok);
                std::string full = tokenizer.decode(generated);
                if (full.size() > printed)
                {
                    std::cout << full.substr(printed) << std::flush;
                    printed = full.size();
                }
                n++;

                std::vector<int> one = {tok};
                tok = model.forward(one);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            std::cout << std::endl << "[生成 " << n << " tokens, " << sec << " s, "
                      << n / std::max(sec, 1e-6) << " tok/s]" << std::endl;

            std::string out = tokenizer.decode(generated);
            out = replace_all(out, IM_END, "");
            out = replace_all(out, EOS_TOK, "");
            out = replace_all(out, BOS_TOK, "");
            first_generation = false;
            return trim(out);
        };

        std::cout << "\n对话开始 (输入 /quit 退出, /reset 清空历史, /system <提示词> 修改系统提示)\n";
        while (true)
        {
            std::cout << "\n>>> " << std::flush;
            std::string line = trim(read_line_utf8());
            if (line.empty() && !std::cin) break;
            if (line.empty()) continue;
            if (line == "/quit" || line == "/exit") break;
            if (line == "/reset")
            {
                history.clear();
                first_generation = true;
                continue;
            }
            if (starts_with(line, "/system "))
            {
                sys_prompt = trim(line.substr(8));
                history.clear();
                first_generation = true;
                std::cout << "[系统提示已更新]" << std::endl;
                continue;
            }
            std::cout << "\n[用户] " << line << std::endl;
            std::cout << "[AstrAI] ";
            std::string reply = run_generation(line);
            if (!reply.empty()) history.emplace_back(line, reply);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "\n错误: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
