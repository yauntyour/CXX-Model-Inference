// 单测: l2norm 全量输出 vs numpy 参考
#include "kernels.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>

static float bf16_to_float(unsigned short v)
{
    unsigned int u = (unsigned int)v << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

int main()
{
    const int n = 49, H = 16, D = 128, C = 6144;
    FILE* f = fopen("d_qkv_conv.bin", "rb");
    if (!f) { printf("no d_qkv_conv.bin\n"); return 2; }
    std::vector<qk::bf16> buf(n * C);
    size_t got = fread(buf.data(), 2, n * C, f);
    fclose(f);
    printf("read %zu elements\n", got);
    printf("host buf[2048..2051]: [");
    for (int i = 2048; i < 2052; i++)
        printf(" %.5f", bf16_to_float(reinterpret_cast<unsigned short&>(buf[i])));
    printf("]\n");

    qk::bf16* dq;
    cudaMalloc(&dq, buf.size() * 2);
    cudaMemcpy(dq, buf.data(), buf.size() * 2, cudaMemcpyHostToDevice);

    // 调试: kernel 前 dump 设备内存 (q+2048 附近) 检查 copy 是否正常
    std::vector<qk::bf16> before(256);
    cudaMemcpy(before.data(), dq + 2048, 256 * 2, cudaMemcpyDeviceToHost);
    FILE* dbg = fopen("dq_before_blk16.bin", "wb");
    fwrite(before.data(), 2, 256, dbg);
    fclose(dbg);
    printf("dq+2048 first8: [");
    for (int i = 0; i < 8; i++)
        printf(" %.5f", bf16_to_float(reinterpret_cast<unsigned short&>(before[i])));
    printf("]\n");
    qk::bf16* q = dq;
    qk::bf16* k = q + (size_t)n * H * D;

    qk::l2norm_qk(q, k, n, H, D);
    cudaDeviceSynchronize();
    cudaMemcpy(buf.data(), dq, buf.size() * 2, cudaMemcpyDeviceToHost);

    FILE* o = fopen("l2_out.bin", "wb");
    fwrite(buf.data(), 2, n * H * D, o);
    fclose(o);
    printf("written l2_out.bin\n");

    // 逐行参考 (numpy 由 python 侧完成; 这里打印前两行的前 8 个值)
    for (int r = 0; r < 2; r++)
    {
        printf("row %d first8: [", r);
        for (int i = 0; i < 8; i++)
            printf(" %.5f", bf16_to_float(reinterpret_cast<unsigned short&>(buf[r * D + i])));
        printf("]\n");
    }
    return 0;
}
