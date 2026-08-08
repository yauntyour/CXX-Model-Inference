#pragma once

#include <vector>
#include <string>
#include <array>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <cctype>
#include <nlohmann/json.hpp>
#include "unicode.hpp"

// ============================================================
// BPE Tokenizer (Qwen2/Qwen3 风格, CPU, 单头文件)
//
// 加载 HuggingFace tokenizer.json, 提供 encode/decode。
// - pretokenize 近似 HF Qwen2Tokenizer 正则 (字母串[含CJK] / 单个数字 / 标点串 /
//   空白 / 换行 / 's 't 等缩写)
// - 特殊 token (如 <|im_start|>、<|image_pad|>、<think>) 按整串匹配输出对应 id
// ============================================================
class BPETokenizer {
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> merge_ranks;
    int bos_id, eos_id;
    std::vector<std::string> specials;

    struct PairHash {
        size_t operator()(const std::pair<std::string,std::string>& p) const {
            return std::hash<std::string>()(p.first) ^
                   (std::hash<std::string>()(p.second) << 1);
        }
    };

    static int utf8_char_len(unsigned char first_byte) {
        if ((first_byte & 0x80) == 0) return 1;
        if ((first_byte & 0xE0) == 0xC0) return 2;
        if ((first_byte & 0xF0) == 0xE0) return 3;
        if ((first_byte & 0xF8) == 0xF0) return 4;
        return 1;
    }

    static int utf8_decode(const std::string& s, size_t& i) {
        if (i >= s.size()) return -1;
        unsigned char c = (unsigned char)s[i];
        int cp = 0, len = utf8_char_len(c);
        if (len == 1) { cp = c; }
        else if (len == 2) { cp = ((c & 0x1F) << 6) | ((unsigned char)s[i+1] & 0x3F); }
        else if (len == 3) { cp = ((c & 0x0F) << 12) | (((unsigned char)s[i+1] & 0x3F) << 6) | ((unsigned char)s[i+2] & 0x3F); }
        else if (len == 4) { cp = ((c & 0x07) << 18) | (((unsigned char)s[i+1] & 0x3F) << 12) | (((unsigned char)s[i+2] & 0x3F) << 6) | ((unsigned char)s[i+3] & 0x3F); }
        i += len;
        return cp;
    }

    static bool is_letter_char(const std::string& utf8_char) {
        if (utf8_char.empty()) return false;
        size_t pos = 0;
        int cp = utf8_decode(utf8_char, pos);
        if (cp < 0) return false;
        return unicode_tbl::is_letter_mark(cp);
    }

    static bool is_digit_char(const std::string& utf8_char) {
        if (utf8_char.empty()) return false;
        size_t pos = 0;
        int cp = utf8_decode(utf8_char, pos);
        if (cp < 0) return false;
        return unicode_tbl::is_number(cp);
    }

    static bool is_space_char(const std::string& utf8_char) {
        if (utf8_char.empty()) return false;
        size_t pos = 0;
        int cp = utf8_decode(utf8_char, pos);
        return cp >= 0 && unicode_tbl::is_whitespace(cp) && cp != '\r' && cp != '\n';
    }

    static bool is_newline_char(const std::string& utf8_char) {
        if (utf8_char.empty()) return false;
        unsigned char c = (unsigned char)utf8_char[0];
        return c == '\r' || c == '\n';
    }

    static constexpr int BL_UNMAPPED = 256;

    static const std::array<uint16_t, 324>& byte_table() {
        static std::array<uint16_t, 324> tbl;
        static bool ready = false;
        if (!ready) {
            tbl.fill((uint16_t)BL_UNMAPPED);
            for (int b = 33; b <= 126; b++) tbl[b] = (uint16_t)b;
            for (int b = 161; b <= 172; b++) tbl[b] = (uint16_t)b;
            for (int b = 174; b <= 255; b++) tbl[b] = (uint16_t)b;
            int n = 0;
            for (int b = 0; b < 256; b++)
                if (tbl[b] == BL_UNMAPPED)
                    tbl[256 + n++] = (uint16_t)b;
            ready = true;
        }
        return tbl;
    }

    static int byte_to_cp(uint8_t b) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
            return b;
        int n = 0;
        for (int x = 0; x < b; x++)
            if (!((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174 && x <= 255)))
                n++;
        return 256 + n;
    }

    static bool is_byte_level_cp(int cp) {
        return cp > 0 && cp <= 323 && byte_table()[cp] != BL_UNMAPPED;
    }

    static uint8_t cp_to_byte(int cp) {
        return (uint8_t)byte_table()[cp];
    }

    static std::string utf8_encode(int cp) {
        std::string s;
        if (cp < 0x80) { s += (char)cp; }
        else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        return s;
    }

    static std::string byte_to_char(uint8_t b) {
        return utf8_encode(byte_to_cp(b));
    }

    int get_merge_rank(const std::string& a, const std::string& b) const {
        auto it = merge_ranks.find(a + " " + b);
        if (it != merge_ranks.end()) return it->second;
        return -1;
    }

    std::vector<std::string> bpe_encode_word(const std::string& word) const {
        std::string bytes_str;
        for (size_t i = 0; i < word.size(); i++) {
            uint8_t b = (uint8_t)word[i];
            bytes_str += byte_to_char(b);
        }

        std::vector<std::string> symbols;
        for (size_t i = 0; i < bytes_str.size(); ) {
            int len = utf8_char_len((unsigned char)bytes_str[i]);
            symbols.push_back(bytes_str.substr(i, len));
            i += len;
        }

        while (symbols.size() > 1) {
            int best_rank = 1 << 30;
            size_t best_idx = 0;
            for (size_t i = 0; i + 1 < symbols.size(); i++) {
                int r = get_merge_rank(symbols[i], symbols[i + 1]);
                if (r >= 0 && r < best_rank) {
                    best_rank = r;
                    best_idx = i;
                }
            }
            if (best_rank == (1 << 30)) break;

            symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1];
            symbols.erase(symbols.begin() + best_idx + 1);
        }
        return symbols;
    }

    // 近似 HF Qwen2 预分词正则:
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    //
    // 关键行为:
    // - 单词可附加一个前导空格 ("hello world" -> "hello" + " world")
    // - 单个标点可附加后续字母串 ("!你好" / ".next" 各自为一块)
    // - 单个数字自成一块 (\p{N})
    std::vector<std::string> pretokenize(const std::string& text) const {
        std::vector<std::string> out;
        size_t n = text.size();
        size_t i = 0;
        // 返回 s[pos-1] 所在字符的起始位置 (UTF-8 回溯)
        auto char_start_before = [&](size_t pos) -> size_t {
            size_t p = pos;
            size_t limit = pos > 4 ? pos - 4 : 0;
            while (p > limit && (((unsigned char)text[p - 1] & 0xC0) == 0x80)) p--;
            return p;
        };
        while (i < n) {
            int len = utf8_char_len((unsigned char)text[i]);
            if (i + len > n) len = (int)(n - i);
            std::string ch = text.substr(i, len);

            if (is_newline_char(ch)) {
                // \s*[\r\n]+ : 吸收前面的空白 + 换行串
                size_t start = i;
                while (start > 0) {
                    size_t cs = char_start_before(start);
                    if (is_space_char(text.substr(cs, start - cs))) start = cs;
                    else break;
                }
                size_t j = i;
                while (j < n) {
                    int l2 = utf8_char_len((unsigned char)text[j]);
                    if (j + l2 > n) break;
                    if (!is_newline_char(text.substr(j, l2))) break;
                    j += l2;
                }
                out.push_back(text.substr(start, j - start));
                i = j;
            } else if (is_space_char(ch)) {
                // 空白串
                size_t j = i;
                while (j < n) {
                    int l2 = utf8_char_len((unsigned char)text[j]);
                    if (j + l2 > n) break;
                    if (!is_space_char(text.substr(j, l2))) break;
                    j += l2;
                }
                size_t run = j - i;
                // 单个空格 + 紧跟字母 -> 附加 (" world")
                if (run == (size_t)len && j < n) {
                    int l2 = utf8_char_len((unsigned char)text[j]);
                    if (j + l2 <= n && is_letter_char(text.substr(j, l2))) {
                        size_t k = j;
                        while (k < n) {
                            int l3 = utf8_char_len((unsigned char)text[k]);
                            if (k + l3 > n) break;
                            if (!is_letter_char(text.substr(k, l3))) break;
                            k += l3;
                        }
                        out.push_back(text.substr(i, k - i));
                        i = k;
                        continue;
                    }
                }
                out.push_back(text.substr(i, j - i));
                i = j;
            } else if (is_digit_char(ch)) {
                out.push_back(ch);
                i += len;
            } else if (is_letter_char(ch)) {
                // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ : 字母串 (含 CJK)
                size_t j = i;
                while (j < n) {
                    int l2 = utf8_char_len((unsigned char)text[j]);
                    if (j + l2 > n) break;
                    if (!is_letter_char(text.substr(j, l2))) break;
                    j += l2;
                }
                out.push_back(text.substr(i, j - i));
                i = j;
            } else {
                // 标点串: 可选单个前导空格 + 非空白/字母/数字串
                size_t start = i;
                // 单个前导空格 (" !" 场景)
                if (i > 0) {
                    size_t cs = char_start_before(i);
                    if (is_space_char(text.substr(cs, i - cs))) {
                        // 仅当该空格前不是空白
                        bool prev_is_space = cs > 0;
                        if (prev_is_space) {
                            size_t cs2 = char_start_before(cs);
                            prev_is_space = is_space_char(text.substr(cs2, cs - cs2));
                        }
                        if (!prev_is_space) start = cs;
                    }
                }
                size_t j = i;
                while (j < n) {
                    int l2 = utf8_char_len((unsigned char)text[j]);
                    if (j + l2 > n) break;
                    std::string c2 = text.substr(j, l2);
                    if (is_space_char(c2) || is_letter_char(c2) || is_digit_char(c2) ||
                        is_newline_char(c2))
                        break;
                    j += l2;
                }
                if (j == i) { out.push_back(ch); i += len; continue; }
                // 单个标点 + 紧跟字母 -> 附加 ("!你好" / ".next")
                bool single = true;
                if (start != i) single = false;  // 带了前导空格则不再附加
                if (single && j - i == (size_t)len && j < n) {
                    int l2 = utf8_char_len((unsigned char)text[j]);
                    if (j + l2 <= n && is_letter_char(text.substr(j, l2))) {
                        size_t k = j;
                        while (k < n) {
                            int l3 = utf8_char_len((unsigned char)text[k]);
                            if (k + l3 > n) break;
                            if (!is_letter_char(text.substr(k, l3))) break;
                            k += l3;
                        }
                        out.push_back(text.substr(start, k - start));
                        i = k;
                        continue;
                    }
                }
                out.push_back(text.substr(start, j - start));
                i = j;
            }
        }
        return out;
    }

    bool is_special(const std::string& s) const {
        return vocab.find(s) != vocab.end() && s.size() >= 2 && s[0] == '<' && s.back() == '>';
    }

    // 文本 -> token ids (不含特殊 token 包装)
    std::vector<int> encode_impl(const std::string& text) const {
        std::vector<int> ids;
        size_t n = text.size();
        size_t i = 0;
        while (i < n) {
            if (text[i] == '<') {
                // 特殊 token 整串匹配: <...>
                size_t close = text.find('>', i + 1);
                if (close != std::string::npos) {
                    std::string cand = text.substr(i, close - i + 1);
                    auto it = vocab.find(cand);
                    if (it != vocab.end()) {
                        ids.push_back(it->second);
                        i = close + 1;
                        continue;
                    }
                }
            }
            // 找到下一个 '<' 位置, 整段预分词 (包含当前 '<' 本身)
            size_t next = text.find('<', i + 1);
            if (next == std::string::npos) next = n;
            std::string seg = text.substr(i, next - i);
            auto words = pretokenize(seg);
            for (auto& w : words) {
                auto syms = bpe_encode_word(w);
                for (auto& s : syms) {
                    auto it = vocab.find(s);
                    if (it != vocab.end()) ids.push_back(it->second);
                }
            }
            i = next;
        }
        return ids;
    }

public:
    BPETokenizer() : bos_id(-1), eos_id(-1) {}

    bool load(const std::string& json_path) {
        std::ifstream f(json_path);
        if (!f.is_open()) return false;

        nlohmann::json j;
        f >> j;

        auto& model = j["model"];
        auto& v = model["vocab"];
        id_to_token.resize(v.size());
        for (auto it = v.begin(); it != v.end(); ++it) {
            std::string token = it.key();
            int id = it.value();
            vocab[token] = id;
            if (id < (int)id_to_token.size())
                id_to_token[id] = token;
        }

        // 特殊 token 不在 model.vocab 中, 需从 added_tokens 加载
        if (j.contains("added_tokens") && j["added_tokens"].is_array())
        {
            for (auto& at : j["added_tokens"])
            {
                std::string token = at["content"].get<std::string>();
                int id = at["id"].get<int>();
                if (vocab.find(token) == vocab.end())
                {
                    vocab[token] = id;
                    if ((int)id_to_token.size() <= id)
                        id_to_token.resize(id + 1);
                    id_to_token[id] = token;
                }
            }
        }

        auto& merges_arr = model["merges"];
        for (size_t i = 0; i < merges_arr.size(); i++) {
            std::string merge_str;
            if (merges_arr[i].is_array())
                merge_str = merges_arr[i][0].get<std::string>() + " " +
                            merges_arr[i][1].get<std::string>();
            else
                merge_str = merges_arr[i].get<std::string>();
            merge_ranks[merge_str] = (int)i;
        }

        auto bos_it = vocab.find("<|im_start|>");
        if (bos_it != vocab.end()) bos_id = bos_it->second;
        auto eos_it = vocab.find("<|im_end|>");
        if (eos_it != vocab.end()) eos_id = eos_it->second;

        // 收集特殊 token (形如 <...> 的 vocab 项)
        for (auto& kv : vocab)
            if (is_special(kv.first))
                specials.push_back(kv.first);

        return true;
    }

    // Encode text. 若 with_special 为 true 则包裹 BOS/EOS (与旧接口兼容).
    std::vector<int> encode(const std::string& text, bool with_special = true) const {
        std::vector<int> ids;
        if (with_special && bos_id >= 0) ids.push_back(bos_id);
        auto inner = encode_impl(text);
        ids.insert(ids.end(), inner.begin(), inner.end());
        if (with_special && eos_id >= 0) ids.push_back(eos_id);
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            if (id < 0 || id >= (int)id_to_token.size()) continue;
            const auto& token = id_to_token[id];
            for (size_t i = 0; i < token.size(); ) {
                int cp = utf8_decode(token, i);
                if (is_byte_level_cp(cp)) {
                    result += (char)cp_to_byte(cp);
                } else if (cp > 0) {
                    result += utf8_encode(cp);
                }
            }
        }
        return result;
    }

    std::string decode_one(int id) const {
        if (id < 0 || id >= (int)id_to_token.size()) return "";
        return decode(std::vector<int>{id});
    }

    int token_id(const std::string& token) const {
        auto it = vocab.find(token);
        return it == vocab.end() ? -1 : it->second;
    }

    size_t vocab_size() const { return id_to_token.size(); }
    int get_bos_id() const { return bos_id; }
    int get_eos_id() const { return eos_id; }
};
