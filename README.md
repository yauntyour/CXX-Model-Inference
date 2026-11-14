<div align="center">

# CXX-Qwen

**纯 C++17 + CUDA 实现的 Qwen3.5-0.8B 多模态推理引擎**

不依赖 PyTorch / TensorFlow，从零构建完整的 LLM 推理系统

[![C++17](https://img.shields.io/badge/C++-17-00599C?style=flat&logo=cplusplus)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-12+-76B900?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 项目简介

CXX-Qwen 是一个完全用 C++17 和 CUDA 编写的 Qwen3.5-0.8B 推理引擎，支持文本生成和视觉理解。它展示了如何在不依赖任何深度学习框架的情况下，从零实现一个完整的多模态大语言模型推理系统。

### 核心特性

- **纯 C++ 实现** — 无 Python 依赖，无 PyTorch/TensorFlow 框架
- **多模态推理** — 支持纯文本和图文混合输入（Vision + Language）
- **CUDA 加速** — 全部推理在 GPU 上完成，BF16 精度
- **混合注意力架构** — 实现 Qwen3.5 独特的 Gated DeltaNet（线性注意力）+ Full Attention 混合架构
- **双模型后端** — 同一套 CUDA 内核同时支持 Qwen3.5 与 Llama 风格 Dense Transformer（AstrAI-V1-2B）
- **高性能推理快路径** — W8A16 GEMV、BF16 WMMA FlashAttention、32-row prefill、split-K decode attention、融合 QKV+RoPE、融合 SwiGLU；AstrAI 2B 实测 decode **145–163 tok/s**（W8A16）、prefill **5.5–9.3k tok/s**；decode 全面领先官方 CUDA，prefill 128/512 领先、2k 持平
- **自研张量库** — 内置 [TensorN](TensorN/) 张量计算库，支持多种后端
- **交互式 CLI** — 支持多轮对话，带 KV 缓存优化

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                      CXX-Qwen 架构                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Tokenizer│  │  Vision  │  │  Qwen3.5 │  │   CUDA   │    │
│  │   (BPE)  │  │ Encoder  │  │  Model   │  │  Kernels │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│        │             │             │             │          │
│        └─────────────┴─────────────┴─────────────┘          │
│                          │                                  │
│                    ┌─────┴─────┐                            │
│                    │  TensorN  │                            │
│                    │  张量库    │                            │
│                    └───────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

### 模型架构 (Qwen3.5-0.8B)

| 参数 | 值 |
|---|---|
| 参数量 | 0.8B |
| 层数 | 24 |
| Hidden Size | 1024 |
| 词表大小 | 248,320 |
| 最大上下文 | 262,144 tokens |
| 注意力类型 | 3× Gated DeltaNet + 1× Full Attention（循环） |
| 位置编码 | mRoPE（多维旋转位置编码） |
| 精度 | BF16 |

### 视觉编码器

| 参数 | 值 |
|---|---|
| 类型 | ViT（Vision Transformer） |
| 层数 | 12 |
| Patch Size | 16×16 |
| Hidden Size | 768 |
| Spatial Merge | 2×2 |

---

## 目录结构

```
CXX-Qwen/
├── CMakeLists.txt              # 顶层构建配置
├── src/                        # 主程序源码
│   ├── main.cpp                # Qwen3.5 CLI 聊天界面入口
│   ├── qwen_model.hpp/cpp      # Qwen3.5 文本模型
│   ├── astrai_main.cpp         # AstrAI / Dense Transformer CLI 入口
│   ├── dense_model.hpp/cpp     # 通用 Dense Transformer（Llama 风格）
│   ├── vision.hpp/cpp          # 视觉编码器
│   ├── image.hpp/cpp           # 图像加载与预处理
│   ├── kernels.hpp/cu          # 基础 CUDA 内核（参考实现）
│   ├── fast_ops.hpp/cu         # 推理快路径：GEMV / 融合算子 / FlashAttention
│   └── tokenizer.hpp           # BPE Tokenizer
├── TensorN/                    # 自研张量计算库
│   ├── core/                   # 核心实现
│   │   ├── tensor.hpp          # N 维张量类
│   │   ├── dtypes.hpp          # 低精度类型支持
│   │   ├── einsum.hpp          # 爱因斯坦求和
│   │   └── CUDA/               # CUDA 后端
│   ├── example/                # 示例程序
│   └── benchmark/              # 性能基准测试
├── tokenizer/                  # Tokenizer 配置文件
└── Qwen3.5-0.8B/              # 模型权重（需单独下载）
```

---

## 快速开始

### 环境要求

- **操作系统**: Windows 10/11 或 Linux
- **编译器**: 支持 C++17 的编译器（MSVC 2019+, GCC 9+, Clang 10+）
- **CUDA**: CUDA Toolkit 12.0+
- **GPU**: NVIDIA GPU（计算能力 7.0+，推荐 RTX 系列）
- **CMake**: 3.18+

### 1. 克隆仓库

```bash
git clone --recursive https://github.com/your-username/CXX-Qwen.git
cd CXX-Qwen
```

### 2. 下载模型权重

从 HuggingFace 下载 Qwen3.5-0.8B 模型：

```bash
# 使用 huggingface-cli
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir Qwen3.5-0.8B

# 或手动下载并放置到 Qwen3.5-0.8B/ 目录
```

所需文件：
- `config.json`
- `model.safetensors-00001-of-00001.safetensors`
- `vocab.json` / `merges.txt`
- `tokenizer.json`（放置到 `tokenizer/` 目录）

### 3. 构建项目

```bash
# MSVC
cmake -B build -G "Visual Studio 18 2026" -A x64 -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_HOST_COMPILER=cl -DCMAKE_CUDA_ARCHITECTURES=120
```

### 4. 运行

```bash
# 文本对话
./qwen35_chat --model ../Qwen3.5-0.8B --tokenizer ../tokenizer/tokenizer.json

# 图文对话
./qwen35_chat --model ../Qwen3.5-0.8B --tokenizer ../tokenizer/tokenizer.json --image photo.jpg

# 性能测试
./qwen35_chat --model ../Qwen3.5-0.8B --tokenizer ../tokenizer/tokenizer.json --selftest --timing
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model <dir>` | 模型权重目录 | `Qwen3.5-0.8B` |
| `--tokenizer <file>` | tokenizer.json 路径 | `tokenizer/tokenizer.json` |
| `--image <file>` | 输入图片路径 | 无（纯文本模式） |
| `--max-new <n>` | 最大生成 token 数 | 512 |
| `--selftest` | 运行自检测试 | 关闭 |
| `--timing` | 显示各阶段耗时 | 关闭 |

---

## 部署 AstrAI-V1-2B-instruct

除 Qwen3.5 外，引擎还内置了一套 **通用 Dense Transformer（Llama 风格）** 后端，
用于运行 [AstrAI-V1-2B-instruct](https://huggingface.co/ViperEkura/AstrAI-V1-2B-instruct)：

| 参数 | 值 |
|---|---|
| 参数量 | 2.0B（BF16） |
| 层数 | 24 |
| Hidden Size | 2048 |
| 注意力 | GQA，16 Q heads / 4 KV heads，head_dim 128 |
| FFN | Dense MLP（8192），SiLU 门控 |
| 归一化 | 标准 RMSNorm（eps=1e-5） |
| 位置编码 | GPT-J 交错式 RoPE（rotary_dim=128，theta=10000） |
| 词表 | 131,072（ByteLevel BPE） |
| 最大上下文 | 32,768 tokens |

### 1. 下载模型

```bash
# 直连 HuggingFace
huggingface-cli download ViperEkura/AstrAI-V1-2B-instruct --local-dir AstrAI-V1-2B-instruct

# 或使用镜像
set HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download ViperEkura/AstrAI-V1-2B-instruct --local-dir AstrAI-V1-2B-instruct
```

目录中需要 `config.json`、`model.safetensors`、`tokenizer.json`。

### 2. 构建

```bash
cmake --build build --config Release --target astrai_chat
```

> 若 CUDA 升级后旧 `build/` 缓存指向已不存在的 Toolkit 路径，可删除
> `build/CMakeCache.txt` 后重新配置，或在配置时指定
> `-DCUDAToolkit_ROOT="<CUDA 安装目录>"`。

### 3. 运行

```bash
# 交互对话
./astrai_chat --model AstrAI-V1-2B-instruct

# 指定上下文长度（默认 8192，最大 32768）
./astrai_chat --model AstrAI-V1-2B-instruct --ctx 16384

# 原始续写 / tokenizer 调试
./astrai_chat --model AstrAI-V1-2B-instruct --probe "Hello"
./astrai_chat --model AstrAI-V1-2B-instruct --encode "你好，世界"

# 吞吐基准（fast 快路径 vs --ref 参考路径）
./astrai_chat --model AstrAI-V1-2B-instruct --bench 64
./astrai_chat --model AstrAI-V1-2B-instruct --bench 64 --ref
```

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model <dir>` | 模型目录 | `AstrAI-V1-2B-instruct` |
| `--tokenizer <file>` | tokenizer.json | `<model>/tokenizer.json` |
| `--system <text>` | 系统提示词 | 内置英文提示 |
| `--max-new <n>` | 最大生成 token 数 | 512 |
| `--ctx <n>` | KV cache 容量 | 8192 |
| `--selftest` | 单 token 自检（含 logits top10） | 关闭 |
| `--encode <text>` | 打印 tokenizer ids 后退出 | 关闭 |
| `--probe <text>` | 原始续写（不走 chat 模板） | 关闭 |
| `--full-prefill` | 每轮全量重算（调试 KV 缓存） | 关闭 |
| `--rope-neox` | 改用 Llama 半维配对 RoPE（兼容其它模型） | 关闭 |
| `--ref` | 关闭快路径，使用参考算子路径 | 关闭 |
| `--bf16` | decode 也使用 BF16 权重（关闭 W8A16） | 关闭（默认 W8A16） |
| `--bench <n>` | 测量 prefill / decode 吞吐 | 关闭 |
| `--timing` | 打印各层耗时 | 关闭 |

> `astrai_chat` 与 `qwen35_chat` 完全独立：前者使用新增的 `dense_model.*`，
> 后者继续使用 Qwen3.5 的 Gated DeltaNet + mRoPE 实现。
>
> 该模型的 RoPE 为 GPT-J 交错式（按固定文本 NLL 验证，1.745 vs Llama 半维配对的
> 2.196）。若运行标准 Llama 配置的模型，可加 `--rope-neox` 切回半维配对。

### 4. 数值验证

仓库提供了 PyTorch 参考实现，用于逐 token 对比：

```bash
# 只看参考实现
python tools/astrai_reference.py --model AstrAI-V1-2B-instruct --prompt "Hello" --max-new 10

# 自动对比 C++ 引擎与 PyTorch: top-k logits、贪心续写
python tools/validate_astrai.py --model AstrAI-V1-2B-instruct --prompt "Hello" --max-new 10

# 通过固定文本 NLL 判定 RoPE / RMSNorm / MLP 激活的配置
python tools/eval_arch.py --model AstrAI-V1-2B-instruct
```

实测结果：`--prompt "The capital of France is"` 下 top-5 全部一致，贪心续写
20 个 token 与 PyTorch 完全一致。

架构判定（固定文本平均 NLL，越低越好）：

| 配置 | mean NLL |
|---|---|
| **GPT-J 交错 RoPE + 标准 RMSNorm + SiLU** | **1.745** |
| Llama 半维 RoPE + 标准 RMSNorm + SiLU | 2.196 |
| GPT-J 交错 RoPE + 标准 RMSNorm + GELU | 2.745 |
| GPT-J 交错 RoPE + `1+w` RMSNorm + SiLU | 13.58 |

### 5. 高性能推理架构

参考 MetaModel 的 fast 层设计，AstrAI 后端默认走 `fast_ops`：

| 模块 | 设计 |
|---|---|
| 权重布局 | 保持 PyTorch 原始 `[out, in]`，**不再做 CPU 转置**；QKV / gate+up 按行合并成一次投影 |
| prefill GEMM | `cublasGemmEx` 直接使用原始 `[out, in]` 权重（cuBLAS `op=T`），bf16 × bf16 → bf16，权重只读一遍 |
| decode GEMV | 自写 warp-per-row GEMV：32 lane 合并读一整行权重，残差加进同一 kernel |
| W8A16 | 默认 decode 使用 int8 权重（per-row symmetric scale）+ BF16 激活，权重流量减半；BF16 路径保留为 `--bf16` |
| QKV/RoPE/KV | 单个融合 kernel：Q/K RoPE + K/V 直接写入 KV cache，省掉 3 次中间拷贝 |
| Prefill 注意力 | BF16 WMMA 闪存注意力：S<256 用 16 行 query（49.8 KB/block，2 blocks/SM）；S≥256 用 32 行 query × 64 key tile、8 warps，K/V tile 每 32 行只读一次 |
| Decode 注意力 | 一个 KV head 一个 block，block 内每个 warp 负责一个 Q head；全程 warp shuffle，无 block sync |
| 长上下文 decode | split-K：每约 128 个 key 切一个 block，各自算局部 `(m, l, o)`，再由 combine kernel 合并；1933-token 上下文把 attention 从 ~21 ms 降到 ~2 ms |
| FFN | gate+up 合并 GEMM + 单 kernel SwiGLU |
| 残差 | o_proj / down_proj 用 `beta=1` 直接累加进 residual，省掉独立的 add kernel 和中间缓冲 |
| 回退 | `--ref` 使用逐算子参考路径，逐 token 对拍用 |

RTX 5060 Ti 16GB 实测（`--bench`，同一权重、同一提示词，GPU 空闲）：

| 上下文 | 场景 | 参考路径 `--ref` | 快路径（默认 W8A16） | 提升 |
|---:|---|---:|---:|---:|
| 70 | prefill | 4,560 tok/s | **6,274 tok/s** | 1.38× |
| 70 | decode | 67.8 tok/s | **161.1 tok/s** | 2.38× |
| 415 | prefill | 4,854 tok/s | **9,125 tok/s** | 1.88× |
| 415 | decode | 45.2 tok/s | **143.5 tok/s** | 3.17× |
| 967 | prefill | 2,830 tok/s | **7,538 tok/s** | 2.66× |
| 967 | decode | 30.5 tok/s | **147.3 tok/s** | 4.83× |
| 1,933 | prefill | 1,597 tok/s | **~5,500 tok/s** | 3.44× |
| 1,933 | decode | 19.0 tok/s | **145.8 tok/s** | 7.67× |
| — | 模型加载 | 10 s（旧转置路径） | **3 s** | 3.3× |

split-K decode 消融（BF16 路径，其余代码完全相同）：

| 上下文 | 单 block decode | split-K decode | 提升 |
|---:|---:|---:|---:|
| 415 | 66.1 tok/s | **94.9 tok/s** | 1.44× |
| 967 | 51.2 tok/s | **93.5 tok/s** | 1.83× |
| 1,933 | 33.2 tok/s | **87.4 tok/s** | 2.65× |

`--skip-attn` 只跑权重 GEMV/MLP 时的权重带宽上限：

| 模式 | decode 下限 | 说明 |
|---|---:|---|
| BF16 | 107 tok/s（9.3 ms/token） | 每 token 约 3.7 GB 权重 |
| W8A16 | **184.6 tok/s（5.4 ms/token）** | 每 token 约 1.9 GB 权重 |

默认 W8A16 的 143–163 tok/s 已接近这个 185 tok/s 的带宽上限，剩余差距主要是
attention 和每 token 的 launch 开销。

> 优化前原始引擎在 142-token 代码生成上约 53 tok/s；当前 W8A16 同负载约
> **145–163 tok/s**，预填充在 128/512 上下文约 9.1–9.5k tok/s。
>
> Qwen3.5-0.8B 的 tied lm_head 与 6 个 full-attention 层的 decode 也复用同一套
> `fo::gemv_bf16` / `fo::decode_attn`；Gated DeltaNet 与视觉编码器仍走原实现。
> Qwen 第二轮（18-token 增量 prefill）warm decode 约 **125 tok/s**。
>
> 调试开关：`FO_NO_SPLITK=1` 关闭 split-K；`FO_GEMV4=1` 切回 4 字节 GEMV；
> `--skip-attn` 跳过注意力，用来单独测量权重带宽上限。

### 6. 与 AstrAI 官方 benchmark 对比

同一份 2B 权重、同一台 RTX 5060 Ti、batch=1。官方框架使用
`repos/AstrAI/scripts/benchmark/inference.py`，CUDA 扩展按 `sm_120a` 编译；
CXX-Qwen 使用 `--bench`。

| 指标 | AstrAI 官方 torch | AstrAI 官方 cuda | CXX BF16 | CXX W8A16 |
|---|---:|---:|---:|---:|
| prefill 128 | 3,691 tok/s | 7,262 tok/s | 8,741 tok/s（139） | **9,263 tok/s** |
| prefill 512 | — | 5,517 tok/s | 9,000 tok/s（553） | **9,300 tok/s** |
| prefill 2,048 | — | **5,586 tok/s** | ~5,500 tok/s（1,933） | ~5,500 tok/s |
| decode @128 | 27.7 ms/step | 9.40 ms（106.4 tok/s） | 10.2 ms（97.9 tok/s） | **6.14 ms（162.8 tok/s）** |
| decode @512 | — | 9.43 ms（106.1 tok/s） | 11.0 ms（91.0 tok/s） | **6.6 ms（~152 tok/s）** |
| decode @2,048 | — | 9.72 ms（102.9 tok/s） | 11.1 ms（~90 tok/s） | **6.86 ms（145.8 tok/s）** |

W8A16 相对 BF16 的 decode 提升：

| 上下文 | BF16 | W8A16 | 提升 |
|---:|---:|---:|---:|
| 139 | 97.9 tok/s | **162.8 tok/s** | 1.66× |
| 553 | 91.0 tok/s | **~152 tok/s** | 1.67× |
| 1,933 | ~90 tok/s | **145.8 tok/s** | 1.62× |

W8A16 质量：`The capital of France is` 上 top-5 全部一致，**20 个贪心 token
与 PyTorch BF16 完全一致**；中文、代码生成样例与 BF16 输出基本一致。
需要严格 BF16 时可加 `--bf16`。W8A16 额外占用约 1.7 GB int8 权重
（BF16 权重保留给 prefill 和回退）。

说明：

- 官方 decode 用 `--kv-windows` 固定 KV 窗口 + CUDA Graph，CUDA event 计时；
  CXX 用 `--bench`，包含模型 forward + argmax，不含 tokenizer。
- 官方 prefill 使用随机 token IDs；CXX 使用真实文本，token 数略有差异。
- 官方框架要求 torch 2.13；本机是 torch 2.8，推理对比时用
  `tools/astrai_stub` 屏蔽了未使用的 Muon 优化器模块。

复现命令（在 `repos/AstrAI` 下）：

```bash
# prefill
python scripts/benchmark/inference.py --ckpt D:/models/AstrAI-V1-2B-instruct \
  --prefill_only --batch_size 1 --prompt_length 512 --num_trials 5 --backend cuda

# 固定 KV 窗口 decode（CUDA Graph + CUDA event）
python scripts/benchmark/inference.py --ckpt D:/models/AstrAI-V1-2B-instruct \
  --decode_only --kv-windows 128,512,2048 --window-steps 32 \
  --batch_size 1 --backend cuda
```

结论：

- **prefill：CXX 32-row FlashAttention 在 128/512 明显领先官方 20%–70%**；
  2,048 长上下文约持平（官方 5,586 vs CXX ~5,500 tok/s，官方 +1–2%）。
- **BF16 decode：官方 CUDA Graph + split-KV 快约 9%–17%**；差距主要是
  kernel launch（约 1 ms）和 2k 上下文的 attention（官方约 0.4 ms，CXX 约 2.1 ms）。
- **W8A16 decode：CXX 领先官方 BF16 约 39%–53%**（145–163 vs 103–106 tok/s），
  且质量对拍 20/20 greedy token 一致。默认模式即 W8A16。
- 同精度 BF16 下，prefill 已经领先/持平官方，decode 仍差 9%–17%；补齐
  **CUDA Graph** 和 **WMMA decode attention** 后可以进一步追平。当前默认
  W8A16 已经在 decode 上形成明显优势。

> 官方自带的 W8A16 `quant_gemm` 是通用 GEMM，M=1 时多数形状并不比 BF16
> 快；把我们的 W8A16 GEMV 单独挂到官方 `F.linear` 上，也会被 PyTorch 的
> per-op/分配开销抵消。要让官方框架达到 CXX 的水平，需要把 **融合 decode
> 路径**（W8A16 GEMV + 融合 QKV/RoPE/KV + split-K attention）搬进官方
> C++ extension，而不是只替换一个 GEMM 算子。

---

## 技术实现

### CUDA 内核

项目实现了约 120 个自定义 CUDA 内核，覆盖 LLM 推理的所有核心算子：

| 类别 | 内核示例 |
|---|---|
| 逐元素 | SiLU, GELU, Sigmoid, Add, Mul |
| 归一化 | RMSNorm, LayerNorm |
| 注意力 | Gated DeltaNet, Full Attention, RoPE |
| 矩阵运算 | GEMM, MatMul, Transpose |
| 视觉 | Patch Embed, Position Embed, Spatial Merge |

### Gated DeltaNet 实现

Qwen3.5 采用创新的混合注意力架构，结合了：

1. **Gated DeltaNet（线性注意力）** — O(n) 复杂度，适合长序列
2. **Full Attention（标准注意力）** — O(n²) 复杂度，捕捉全局依赖

```cpp
// 简化的 Gated DeltaNet 前向传播
void gated_delta_rule_forward(
    const bf16* Q, const bf16* K, const bf16* V,
    bf16* O, float* S,  // 状态矩阵
    int n_heads, int seq_len, int head_dim
) {
    // 1. 计算门控参数 g
    // 2. 更新状态 S = S * diag(exp(g)) + K^T V
    // 3. 输出 O = Q * S
}
```

---

## TensorN 张量库

CXX-Qwen 内置了 [TensorN](TensorN/) 张量计算库，这是一个功能完整的 C++17 张量库：

### 核心功能

- **多后端支持**: 原生 C++ / OpenBLAS / CUDA cuBLAS
- **低精度类型**: FP16, BF16, TF32, FP8 (E4M3/E5M2)
- **爱因斯坦求和**: 类似 NumPy 的 `einsum` 操作
- **数据 I/O**: CSV, NumPy (.npy/.npz), JSON, PyTorch (.pt), GGUF, safetensors

### 性能基准

在 RTX 5060 Ti 上的矩阵乘法性能：

| 后端 | 1024×1024 矩阵乘法 | 相对速度 |
|---|---|---|
| 原生 C++ | ~2.5s | 1× |
| OpenBLAS | ~0.008s | 312× |
| CUDA cuBLAS | ~0.000009s | 277,778× |

更多详情请参阅 [TensorN 文档](TensorN/readme.md)。

---

## 示例

### 纯文本对话

```
User: 你好，请介绍一下自己。

Assistant: 你好！我是通义千问（Qwen），一个由阿里云开发的AI助手。
我可以帮助你回答问题、进行对话交流、提供建议等。有什么我可以帮助你的吗？
```

### 图文理解

```
User: [图片] 这张图片里有什么？

Assistant: 这张图片展示了一只可爱的橘色猫咪，它正趴在窗台上晒太阳。
猫咪的眼睛半闭着，看起来非常惬意和放松。阳光洒在它的毛发上，
呈现出温暖的金色光泽。
```

---

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

模型权重来自 [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)，受 [Apache 2.0 许可证](Qwen3.5-0.8B/LICENSE) 约束。

---

## 致谢

- [Qwen Team](https://github.com/QwenLM) — 提供 Qwen3.5 模型
- [stb](https://github.com/nothings/stb) — 单头文件图像加载库
- [nlohmann/json](https://github.com/nlohmann/json) — JSON 解析库

---

<div align="center">

**[English](TensorN/docx/readme_en.md) · 中文**

</div>
