// ============================================================
// 图像加载与预处理实现
// ============================================================

#include "image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cmath>
#include <stdexcept>

bool load_image(const std::string& path, LoadedImage& img)
{
    int w = 0, h = 0, ch = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, 3);
    if (!data) return false;
    img.w = w;
    img.h = h;
    // stb 为 [H, W, C] 行主序; 转成 [C, H, W] (后续 bicubic/prep 均按通道主序)
    img.rgb.resize((size_t)3 * w * h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int c = 0; c < 3; c++)
                img.rgb[(size_t)c * w * h + (size_t)y * w + x] = data[(y * w + x) * 3 + c] / 255.0f;
    stbi_image_free(data);
    return true;
}

void smart_resize(int height, int width, int factor, int min_pixels, int max_pixels,
                  int& out_h, int& out_w)
{
    auto round_to = [&](int v) { return (int)std::round((double)v / factor) * factor; };
    int h_bar = round_to(height);
    int w_bar = round_to(width);
    if (h_bar * w_bar > max_pixels)
    {
        double beta = std::sqrt((double)height * width / max_pixels);
        h_bar = (int)std::max(factor, (int)std::floor(height / beta / factor) * factor);
        w_bar = (int)std::max(factor, (int)std::floor(width / beta / factor) * factor);
    }
    else if (h_bar * w_bar < min_pixels)
    {
        double beta = std::sqrt((double)min_pixels / (height * width));
        h_bar = (int)std::ceil(height * beta / factor) * factor;
        w_bar = (int)std::ceil(width * beta / factor) * factor;
    }
    out_h = h_bar;
    out_w = w_bar;
}

namespace
{
    // 双三次插值 (Keys a=-0.75, 与 torchvision BICUBIC 一致), 无抗锯齿
    inline void bicubic_taps(double src, double dst_size, double src_size, int& i0, int& i1,
                             int& i2, int& i3, double& w0, double& w1, double& w2, double& w3)
    {
        double coord = (src + 0.5) * src_size / dst_size - 0.5;
        int f = (int)std::floor(coord);
        double t = coord - f;
        const double a = -0.75;
        double w[4];
        for (int i = 0; i < 4; i++)
        {
            double d = std::abs(t - (i - 1));
            double val;
            if (d <= 1.0)
                val = ((a + 2) * d - (a + 3)) * d * d + 1;
            else if (d < 2.0)
                val = ((a * d - 5 * a) * d + 8 * a) * d - 4 * a;
            else
                val = 0.0;
            w[i] = val;
        }
        int taps[4] = {f - 1, f, f + 1, f + 2};
        for (int i = 0; i < 4; i++)
        {
            if (taps[i] < 0) taps[i] = 0;
            if (taps[i] >= (int)src_size) taps[i] = (int)src_size - 1;
        }
        i0 = taps[0]; i1 = taps[1]; i2 = taps[2]; i3 = taps[3];
        w0 = w[0]; w1 = w[1]; w2 = w[2]; w3 = w[3];
    }

    // 分离式双三次缩放: rgb [3, H, W] (值域 [0,1]) -> [3, Hr, Wr]
    void bicubic_resize(const std::vector<float>& src, int H, int W, int Hr, int Wr,
                        std::vector<float>& dst)
    {
        dst.assign((size_t)3 * Hr * Wr, 0.0f);
        // 水平
        std::vector<float> tmp((size_t)3 * H * Wr);
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < Wr; x++)
                {
                    int i0, i1, i2, i3;
                    double w0, w1, w2, w3;
                    bicubic_taps(x, Wr, W, i0, i1, i2, i3, w0, w1, w2, w3);
                    const float* row = &src[(size_t)c * H * W + (size_t)y * W];
                    double v = w0 * row[i0] + w1 * row[i1] + w2 * row[i2] + w3 * row[i3];
                    tmp[(size_t)c * H * Wr + (size_t)y * Wr + x] = (float)v;
                }
        // 垂直
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < Hr; y++)
            {
                int i0, i1, i2, i3;
                double w0, w1, w2, w3;
                bicubic_taps(y, Hr, H, i0, i1, i2, i3, w0, w1, w2, w3);
                for (int x = 0; x < Wr; x++)
                {
                    double v = w0 * tmp[(size_t)c * H * Wr + (size_t)i0 * Wr + x] +
                               w1 * tmp[(size_t)c * H * Wr + (size_t)i1 * Wr + x] +
                               w2 * tmp[(size_t)c * H * Wr + (size_t)i2 * Wr + x] +
                               w3 * tmp[(size_t)c * H * Wr + (size_t)i3 * Wr + x];
                    dst[(size_t)c * Hr * Wr + (size_t)y * Wr + x] = (float)v;
                }
            }
    }
}

VisionPrep prepare_image(const LoadedImage& img)
{
    const int patch = 16;
    const int merge = 2;
    const int factor = patch * merge;
    const int min_pixels = 65536;      // 256^2
    const int max_pixels = 16777216;   // 4096^2

    int Hr, Wr;
    smart_resize(img.h, img.w, factor, min_pixels, max_pixels, Hr, Wr);

    std::vector<float> resized;
    bicubic_resize(img.rgb, img.h, img.w, Hr, Wr, resized);

    const int grid_h = Hr / patch;   // 特征网格
    const int grid_w = Wr / patch;
    const int P = grid_h * grid_w;   // 总 patch 数

    VisionPrep prep;
    prep.grid_h = grid_h;
    prep.grid_w = grid_w;
    prep.n_patches = P;
    prep.n_tokens = P / (merge * merge);

    // 归一化 (mean=std=0.5) 并构成双帧 [3, 2, H, W]
    prep.pixels.resize((size_t)3 * 2 * Hr * Wr);
    for (int c = 0; c < 3; c++)
        for (int f = 0; f < 2; f++)
            for (int y = 0; y < Hr; y++)
                for (int x = 0; x < Wr; x++)
                {
                    float v = resized[(size_t)c * Hr * Wr + (size_t)y * Wr + x];
                    prep.pixels[((size_t)c * 2 + f) * Hr * Wr + (size_t)y * Wr + x] =
                        (v - 0.5f) / 0.5f;
                }

    // 块主序 (spatial merge 2x2): 与 HF 一致
    // - patch_order[r] = 光栅 patch r 的块主序位置 (patch_embed 输出打散)
    // - interp_idx/interp_wgt/vis_pos 均按块主序索引
    const int side = 48;  // num_grid_per_side = sqrt(2304)
    prep.patch_order.resize(P);
    prep.interp_idx.assign((size_t)P * 4, 0);
    prep.interp_wgt.assign((size_t)P * 4, 0.0f);
    prep.vis_pos.assign((size_t)P * 2, 0);
    for (int r = 0; r < P; r++)
    {
        int row = r / grid_w, col = r % grid_w;
        int bm = ((row / merge) * (grid_w / merge) + col / merge) * 4 +
                 (row % merge) * 2 + (col % merge);
        prep.patch_order[r] = bm;
    }
    for (int p = 0; p < P; p++)
    {
        // 由块主序索引解码 (row, col)
        int in_col = p % merge;
        int in_row = (p / merge) % merge;
        int block_col = (p / (merge * merge)) % (grid_w / merge);
        int block_row = p / (merge * merge * (grid_w / merge));
        int row = block_row * merge + in_row;
        int col = block_col * merge + in_col;
        prep.vis_pos[2 * p + 0] = row;
        prep.vis_pos[2 * p + 1] = col;
        // 双线性插值 pos_embed (align_corners=True, side=48)
        double srow = (double)row * (side - 1) / std::max(grid_h - 1, 1);
        double scol = (double)col * (side - 1) / std::max(grid_w - 1, 1);
        int fr = (int)std::floor(srow), fc = (int)std::floor(scol);
        int tr[2] = {fr, fr + 1};
        int tc[2] = {fc, fc + 1};
        if (tr[1] > side - 1) tr[1] = side - 1;
        if (tc[1] > side - 1) tc[1] = side - 1;
        double wr0 = 1.0 - (srow - fr), wr1 = 1.0 - wr0;
        double wc0 = 1.0 - (scol - fc), wc1 = 1.0 - wc0;
        int n = 0;
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++)
            {
                prep.interp_idx[p * 4 + n] = tr[a] * side + tc[b];
                prep.interp_wgt[p * 4 + n] = (float)((a == 0 ? wr0 : wr1) * (b == 0 ? wc0 : wc1));
                n++;
            }
    }
    return prep;
}
