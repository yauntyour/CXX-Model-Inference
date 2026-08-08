#pragma once
// ============================================================
// 图像加载 (stb_image) 与 Qwen3.5 预处理
// (smart_resize + 双三次缩放 + 归一化 + 块主序索引)
// ============================================================

#include "vision.hpp"

#include <string>
#include <vector>

// 解码后的图像 (RGB, 归一化到 [0,1] 尺度再缩放)
struct LoadedImage
{
    int w = 0, h = 0;
    std::vector<float> rgb;   // [3, H, W], 值域 [0,1]
};

bool load_image(const std::string& path, LoadedImage& img);

// 按 HF Qwen2VL smart_resize: 尺寸为 factor=32 的倍数, 像素数在 [min,max]
void smart_resize(int height, int width, int factor, int min_pixels, int max_pixels,
                  int& out_h, int& out_w);

// 完整预处理: 缩放 + 归一化 + patch 索引 (供 VisionModel::forward)
VisionPrep prepare_image(const LoadedImage& img);
