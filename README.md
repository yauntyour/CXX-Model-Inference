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
│   ├── main.cpp                # CLI 聊天界面入口
│   ├── qwen_model.hpp/cpp      # Qwen3.5 文本模型
│   ├── vision.hpp/cpp          # 视觉编码器
│   ├── image.hpp/cpp           # 图像加载与预处理
│   ├── kernels.hpp/cu          # 自定义 CUDA 内核
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
# 创建构建目录
mkdir build && cd build

# 配置（可指定 CUDA 架构）
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86  # RTX 3060 = 86, RTX 4090 = 89

# 编译
cmake --build . --config Release
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
