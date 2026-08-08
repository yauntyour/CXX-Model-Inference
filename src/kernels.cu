// ============================================================
// Qwen3.5 推理自定义 CUDA 内核
// 存储与逐元素运算直接用 bf16 (__nv_bfloat16 原生算术);
// 规约类 (softmax / norm 方差 / 注意力 / gated delta) 按 HF 参考实现
// 在 fp32 中累加, 结果回写 bf16。
// ============================================================

#include "kernels.hpp"

#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdio>
#include <stdexcept>

namespace qk
{

    __device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
    __device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }

    // ---------------- block reduce helpers ----------------
    // 注意: 末级跨 warp 归约必须保持整 warp 收敛,
    // 否则 __shfl_down_sync(0xffffffff, ...) 在部分 lane 发散时会死锁

    __device__ __forceinline__ float warp_reduce(float v)
    {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        return v;
    }

    // 全 block 归约 (blockDim 为 32 的倍数, ≤1024)
    __device__ __forceinline__ void block_reduce(float& v, float* shared, int tid,
                                                 int blockDimX)
    {
        v = warp_reduce(v);
        if ((tid & 31) == 0) shared[tid >> 5] = v;
        __syncthreads();
        int nwarp = blockDimX >> 5;
        if (tid < 32)   // warp 0 全部 32 条 lane 活跃, 无发散
        {
            float w = (tid < nwarp) ? shared[tid] : 0.f;
            w = warp_reduce(w);
            if (tid == 0) shared[0] = w;
        }
        __syncthreads();
        v = shared[0];
    }

    // ---------------- 逐元素 (原生 bf16 算术) ----------------

    __global__ void silu_kernel(const bf16* x, bf16* y, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            bf16 v = x[i];
            y[i] = v * hrcp(bf16(1.f) + hexp(-v));
        }
    }
    void silu(const bf16* x, bf16* y, int n, cudaStream_t s)
    {
        silu_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    }

    __global__ void sigmoid_kernel(const bf16* x, bf16* y, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = hrcp(bf16(1.f) + hexp(-x[i]));
    }
    void sigmoid(const bf16* x, bf16* y, int n, cudaStream_t s)
    {
        sigmoid_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    }

    __global__ void gelu_tanh_kernel(const bf16* x, bf16* y, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            bf16 v = x[i];
            bf16 inner = bf16(0.7978845608028654f) * (v + bf16(0.044715f) * v * v * v);
            y[i] = bf16(0.5f) * v * (bf16(1.f) + htanh(inner));
        }
    }
    void gelu_tanh(const bf16* x, bf16* y, int n, cudaStream_t s)
    {
        gelu_tanh_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    }

    // 精确 GELU (nn.GELU() 默认, 视觉 merger 用; 与 gelu_pytorch_tanh 不同)
    __global__ void gelu_kernel(const bf16* x, bf16* y, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            float v = b2f(x[i]);
            y[i] = f2b(0.5f * v * (1.f + erff(v * 0.7071067811865476f)));
        }
    }
    void gelu(const bf16* x, bf16* y, int n, cudaStream_t s)
    {
        gelu_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    }

    __global__ void sigmoid_mul_kernel(const bf16* a, const bf16* g, bf16* y, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a[i] * hrcp(bf16(1.f) + hexp(-g[i]));
    }
    void sigmoid_mul(const bf16* a, const bf16* g, bf16* y, int n, cudaStream_t s)
    {
        sigmoid_mul_kernel<<<(n + 255) / 256, 256, 0, s>>>(a, g, y, n);
    }

    __global__ void add_inplace_kernel(bf16* x, const bf16* y, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) x[i] = x[i] + y[i];
    }
    void add_inplace(bf16* x, const bf16* y, int n, cudaStream_t s)
    {
        add_inplace_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    }

    __global__ void mul_inplace_kernel(bf16* x, const bf16* y, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) x[i] = x[i] * y[i];
    }
    void mul_inplace(bf16* x, const bf16* y, int n, cudaStream_t s)
    {
        mul_inplace_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    }

    // y = x + bias (每行加偏置); d 可能 > 1024 (如视觉 qkv 2304), 用循环
    __global__ void add_bias_kernel(const bf16* x, const bf16* bias, bf16* y, int n, int d)
    {
        int row = blockIdx.x;
        int i = threadIdx.x;
        for (; i < d; i += blockDim.x)
            y[(size_t)row * d + i] = x[(size_t)row * d + i] + bias[i];
    }
    void add_bias(bf16* x, const bf16* bias, bf16* y, int n, int d, cudaStream_t s)
    {
        add_bias_kernel<<<n, 256, 0, s>>>(x, bias, y, n, d);
    }

    // g = -exp(A_log) * softplus(a + dt_bias), fp32 计算 (与 HF 一致); A_log: [H] fp32
    __global__ void gated_delta_params_kernel(const bf16* a, const float* A_log,
                                              const bf16* dt_bias, bf16* g, int n, int H)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        int h = i % H;
        float x = b2f(a[i]) + b2f(dt_bias[h]);
        float sp = x > 20.f ? x : log1pf(__expf(x));
        g[i] = f2b(-__expf(A_log[h]) * sp);
    }
    void gated_delta_params(const bf16* a, const float* A_log, const bf16* dt_bias,
                            bf16* g, int n, int H, cudaStream_t s)
    {
        gated_delta_params_kernel<<<(n + 255) / 256, 256, 0, s>>>(a, A_log, dt_bias, g, n, H);
    }

    // ---------------- 归一化 (方差/均值 fp32 累加, 与 HF 一致) ----------------

    template <int DMAX>
    __global__ void rms_norm_1pw_kernel(const bf16* x, const bf16* w, bf16* y, int n, int d,
                                        float eps)
    {
        int row = blockIdx.x;
        const bf16* xr = x + (size_t)row * d;
        bf16* yr = y + (size_t)row * d;
        int tid = threadIdx.x;
        __shared__ float ssum[DMAX / 8 + 1];
        float v = 0.f;
        for (int i = tid; i < d; i += blockDim.x)
        {
            float t = b2f(xr[i]);
            v += t * t;
        }
        block_reduce(v, ssum, tid, blockDim.x);
        float mean = __frsqrt_rn(v / (float)d + eps);
        for (int i = tid; i < d; i += blockDim.x)
            yr[i] = f2b(b2f(xr[i]) * mean * (1.f + b2f(w[i])));
    }
    void rms_norm_1pw(const bf16* x, const bf16* w, bf16* y, int n, int d, float eps, cudaStream_t s)
    {
        rms_norm_1pw_kernel<2048><<<n, 256, 0, s>>>(x, w, y, n, d, eps);
    }

    __global__ void rms_norm_gated_kernel(const bf16* x, const bf16* w, const bf16* gate,
                                          bf16* y, int n, int d, float eps)
    {
        int row = blockIdx.x;
        const bf16* xr = x + (size_t)row * d;
        const bf16* gr = gate + (size_t)row * d;
        bf16* yr = y + (size_t)row * d;
        int tid = threadIdx.x;
        __shared__ float ssum[256 / 8 + 1];
        float v = 0.f;
        for (int i = tid; i < d; i += blockDim.x)
        {
            float t = b2f(xr[i]);
            v += t * t;
        }
        block_reduce(v, ssum, tid, blockDim.x);
        float mean = __frsqrt_rn(v / (float)d + eps);
        for (int i = tid; i < d; i += blockDim.x)
        {
            float gv = b2f(gr[i]);
            float sil = gv / (1.f + __expf(-gv));
            yr[i] = f2b(b2f(xr[i]) * mean * b2f(w[i]) * sil);
        }
    }
    void rms_norm_gated(const bf16* x, const bf16* w, const bf16* gate, bf16* y,
                        int n, int d, float eps, cudaStream_t s)
    {
        rms_norm_gated_kernel<<<n, 256, 0, s>>>(x, w, gate, y, n, d, eps);
    }

    __global__ void layer_norm_kernel(const bf16* x, const bf16* w, const bf16* b, bf16* y,
                                      int n, int d, float eps)
    {
        int row = blockIdx.x;
        const bf16* xr = x + (size_t)row * d;
        bf16* yr = y + (size_t)row * d;
        int tid = threadIdx.x;
        __shared__ float ssum[256 / 8 + 1];
        float sum = 0.f, sq = 0.f;
        for (int i = tid; i < d; i += blockDim.x)
        {
            float t = b2f(xr[i]);
            sum += t;
            sq += t * t;
        }
        block_reduce(sum, ssum, tid, blockDim.x);
        float sqsum = sq;
        block_reduce(sqsum, ssum, tid, blockDim.x);
        float mean = sum / (float)d;
        float v = sqsum / (float)d - mean * mean;
        __shared__ float smean, sstd;
        if (tid == 0) { smean = mean; sstd = __frsqrt_rn(v + eps); }
        __syncthreads();
        for (int i = tid; i < d; i += blockDim.x)
            yr[i] = f2b((b2f(xr[i]) - smean) * sstd * b2f(w[i]) + b2f(b[i]));
    }
    void layer_norm(const bf16* x, const bf16* w, const bf16* b, bf16* y,
                    int n, int d, float eps, cudaStream_t s)
    {
        layer_norm_kernel<<<n, 256, 0, s>>>(x, w, b, y, n, d, eps);
    }

    // ---------------- embedding gather ----------------

    __global__ void embed_gather_kernel(const bf16* table, const int* ids, bf16* out,
                                        int n, int d)
    {
        int row = blockIdx.x;
        int t = threadIdx.x;
        int id = ids[row];
        if (id < 0) return;
        const bf16* e = table + (size_t)id * d;
        out[(size_t)row * d + t] = e[t];
    }
    void embed_gather(const bf16* table, const int* ids, bf16* out, int n, int d, cudaStream_t s)
    {
        embed_gather_kernel<<<n, 1024, 0, s>>>(table, ids, out, n, d);
    }

    // ---------------- q/gate 拆分 ----------------
    // HF 布局: q_proj 输出每 token 内按头交错 [q0 g0 q1 g1 ... q(H-1) g(H-1)],
    // 每头占 2*D (前 D 为 q, 后 D 为 gate)。拆成连续 q/gate: [n, H, D]

    __global__ void split_qgate_kernel(const bf16* x, bf16* q, bf16* gate,
                                       int n, int H, int D)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n * H * D) return;
        int hd = idx % (H * D);
        int t = idx / (H * D);
        int h = hd / D;
        int d = hd % D;
        const bf16* row = x + (size_t)t * H * 2 * D;
        q[idx] = row[h * 2 * D + d];
        gate[idx] = row[h * 2 * D + D + d];
    }
    void split_qgate(const bf16* x, bf16* q, bf16* gate, int n, int H, int D,
                     cudaStream_t s)
    {
        split_qgate_kernel<<<(n * H * D + 255) / 256, 256, 0, s>>>(x, q, gate, n, H, D);
    }

    // ---------------- RoPE 文本 (mrope, 部分旋转) ----------------
    // 与 HF apply_interleaved_mrope 一致: 对旋转对 j, 通道 g = j % 3
    // (j < 3*section[g] 时用对应通道位置, 否则回退 T), 频率 = inv_freq[j] (直接用 j)。
    // 位置与频率在 fp32 中计算, cos/sin 回写 bf16

    __global__ void rope_text_kernel(const bf16* x, bf16* y,
                                     const int* pos_t, const int* pos_h, const int* pos_w,
                                     const float* inv_freq, const int* section,
                                     int n, int H, int D, int rot_dim)
    {
        int t = threadIdx.x;
        int blk = blockIdx.x;            // 每个 token 一个 block
        int h = t / 32;
        int j = t % 32;                  // 旋转对索引 [0, 32)
        if (h >= H) return;
        int p = pos_t[blk];
        int g = j % 3;
        if (g == 1 && j < section[1] * 3) p = pos_h[blk];
        else if (g == 2 && j < section[2] * 3) p = pos_w[blk];
        float m = (float)p * inv_freq[j];
        float c = __cosf(m), s = __sinf(m);
        const bf16* xr = x + ((size_t)blk * H + h) * D;
        bf16* yr = y + ((size_t)blk * H + h) * D;
        if (j < rot_dim / 2)
        {
            float qr = b2f(xr[j]);
            float qi = b2f(xr[j + rot_dim / 2]);
            yr[j] = f2b(qr * c - qi * s);
            yr[j + rot_dim / 2] = f2b(qi * c + qr * s);
        }
        // 未旋转维 [rot_dim, D): 每头 32 线程按 j 步进 (不能用 t, 否则 H < 8 时
        // (如 K/V, Hkv=2) 只拷贝 [rot_dim, 32+rot_dim), 后半段残留脏数据)
        for (int d2 = rot_dim + j; d2 < D; d2 += 32) yr[d2] = xr[d2];
    }
    void rope_text(const bf16* x, bf16* y, const int* pos_t, const int* pos_h, const int* pos_w,
                   const float* inv_freq, const int* section, int n, int H, int D, int rot_dim,
                   cudaStream_t s)
    {
        rope_text_kernel<<<n, 256, 0, s>>>(x, y, pos_t, pos_h, pos_w, inv_freq, section,
                                           n, H, D, rot_dim);
    }

    // ---------------- RoPE 视觉 (全维旋转) ----------------

    __global__ void rope_vision_kernel(const bf16* x, bf16* y, const int* pos,
                                       const float* inv_freq, int n, int H, int D,
                                       int x_stride, int y_stride)
    {
        int t = threadIdx.x;
        int blk = blockIdx.x;
        int h = t / (D / 2);
        int j = t % (D / 2);
        if (h >= H) return;
        int c = j / 16;                 // 0=H, 1=W
        int k = j % 16;
        float m = (float)pos[blk * 2 + c] * inv_freq[k];
        float cc = __cosf(m), ss = __sinf(m);
        const bf16* xr = x + (size_t)blk * x_stride + h * D;
        bf16* yr = y + (size_t)blk * y_stride + h * D;
        float qr = b2f(xr[j]);
        float qi = b2f(xr[j + D / 2]);
        yr[j] = f2b(qr * cc - qi * ss);
        yr[j + D / 2] = f2b(qi * cc + qr * ss);
    }
    void rope_vision(const bf16* x, bf16* y, const int* pos, const float* inv_freq,
                     int n, int H, int D, int x_stride, int y_stride, cudaStream_t s)
    {
        rope_vision_kernel<<<n, H * (D / 2), 0, s>>>(x, y, pos, inv_freq, n, H, D,
                                                      x_stride, y_stride);
    }

    // ---------------- 因果 depthwise conv1d (bf16 乘累加) ----------------

    __global__ void conv1d_causal_kernel(const bf16* x, const bf16* w, bf16* state, bf16* y,
                                         int seq, int C, int K)
    {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= C) return;
        const bf16* wc = w + c * K;
        bf16* st = state + c * (K - 1);
        bf16 ring[4];
        for (int k = 0; k < K - 1; k++) ring[k] = st[k];
        for (int t = 0; t < seq; t++)
        {
            bf16 xin = x[(size_t)t * C + c];
            float acc = 0.f;
            for (int k = 0; k < K - 1; k++) acc += b2f(wc[k]) * b2f(ring[k]);
            acc += b2f(wc[K - 1]) * b2f(xin);
            bf16 v = f2b(acc);
            y[(size_t)t * C + c] = v * hrcp(bf16(1.f) + hexp(-v));
            for (int k = 0; k < K - 2; k++) ring[k] = ring[k + 1];
            ring[K - 2] = xin;
        }
        for (int k = 0; k < K - 1; k++) st[k] = ring[k];
    }
    void conv1d_causal(const bf16* x, const bf16* w, bf16* state, bf16* y,
                       int seq, int C, int K, cudaStream_t s)
    {
        conv1d_causal_kernel<<<(C + 255) / 256, 256, 0, s>>>(x, w, state, y, seq, C, K);
    }

    __global__ void conv1d_step_kernel(const bf16* x, const bf16* w, bf16* state, bf16* y,
                                       int C, int K)
    {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= C) return;
        const bf16* wc = w + c * K;
        bf16* st = state + c * (K - 1);
        float acc = 0.f;
        for (int k = 0; k < K - 1; k++) acc += b2f(wc[k]) * b2f(st[k]);
        bf16 xin = x[c];
        acc += b2f(wc[K - 1]) * b2f(xin);
        bf16 v = f2b(acc);
        y[c] = v * hrcp(bf16(1.f) + hexp(-v));
        for (int k = 0; k < K - 2; k++) st[k] = st[k + 1];
        st[K - 2] = xin;
    }
    void conv1d_causal_step(const bf16* x, const bf16* w, bf16* state, bf16* y,
                            int C, int K, cudaStream_t s)
    {
        conv1d_step_kernel<<<(C + 255) / 256, 256, 0, s>>>(x, w, state, y, C, K);
    }

    // ---------------- q/k l2norm (fp32 累加) ----------------
    // 行主序 [n, stride], q 列 0, k 列 off (off = H*D)

    __global__ void l2norm_qk_kernel(bf16* q, bf16* k, int n, int H, int D, int off, int stride)
    {
        int bh = blockIdx.x;
        int h = bh % H;
        int row = bh / H;
        int t = threadIdx.x;
        bf16* qr = q + (size_t)row * stride + h * D;
        bf16* kr = k + (size_t)row * stride + h * D;
        __shared__ float sq[256 / 8 + 1], skk[256 / 8 + 1];
        float vq = 0.f, vk = 0.f;
        for (int i = t; i < D; i += blockDim.x)
        {
            float a = b2f(qr[i]);
            float b = b2f(kr[i]);
            vq += a * a;
            vk += b * b;
        }
        block_reduce(vq, sq, t, blockDim.x);
        block_reduce(vk, skk, t, blockDim.x);
        float iq = __frsqrt_rn(vq + 1e-6f);
        float ik = __frsqrt_rn(vk + 1e-6f);
        float qscale = iq * __frsqrt_rn((float)D);
        for (int i = t; i < D; i += blockDim.x)
        {
            qr[i] = f2b(b2f(qr[i]) * qscale);
            kr[i] = f2b(b2f(kr[i]) * ik);
        }
    }
    void l2norm_qk(bf16* q, bf16* k, int n, int H, int D, int off, int stride, cudaStream_t s)
    {
        l2norm_qk_kernel<<<n * H, 256, 0, s>>>(q, k, n, H, D, off, stride);
    }

    // ---------------- Gated Delta Rule (fp32 累加, 与 HF 一致) ----------------
    // 行主序 [n, stride]: q 列 0, k 列 off, v 列 2*off (off = H*D)

    // 动态共享内存布局: q[128] + k[128] + v[128] + state[128*128]
    __global__ void gated_delta_kernel(const bf16* q, const bf16* k, const bf16* v,
                                       const bf16* beta, const bf16* g,
                                       float* state, bf16* out, int seq, int H, int D,
                                       int stride)
    {
        extern __shared__ float smem[];
        float* sq = smem;
        float* sk = smem + D;
        float* sv = smem + 2 * D;
        float* sst = smem + 3 * D;
        int h = blockIdx.x;
        int t = threadIdx.x;
        float* gst = state + (size_t)h * D * D;

        for (int i = t; i < D * D; i += blockDim.x) sst[i] = gst[i];
        __syncthreads();

        for (int seq_i = 0; seq_i < seq; seq_i++)
        {
            const bf16* qr = q + (size_t)seq_i * stride + h * D;
            const bf16* kr = k + (size_t)seq_i * stride + h * D;
            const bf16* vr = v + (size_t)seq_i * stride + h * D;
            if (t < D)
            {
                sq[t] = b2f(qr[t]);
                sk[t] = b2f(kr[t]);
                sv[t] = b2f(vr[t]);
            }
            __syncthreads();

            float decay = __expf(b2f(g[seq_i * H + h]));
            float bet = b2f(beta[seq_i * H + h]);

            // 与 HF 参考一致: state 先衰减, 再按 delta rule 更新,
            // 输出用更新后的 state 计算: out_t = (g*S + k⊗delta)·q
            // (j = t, 本线程只读写自己所属的列, 无跨线程竞争)
            for (int k = 0; k < D; k++) sst[k * D + t] *= decay;

            float kv = 0.f;
            for (int k = 0; k < D; k++) kv += sst[k * D + t] * sk[k];
            float d = (sv[t] - kv) * bet;
            for (int k = 0; k < D; k++) sst[k * D + t] += sk[k] * d;

            float o = 0.f;
            for (int k = 0; k < D; k++) o += sst[k * D + t] * sq[k];
            out[(size_t)seq_i * H * D + h * D + t] = f2b(o);

            __syncthreads();
        }
        for (int i = t; i < D * D; i += blockDim.x) gst[i] = sst[i];
    }
    void gated_delta(const bf16* q, const bf16* k, const bf16* v, const bf16* beta,
                     const bf16* g, float* state, bf16* out, int seq, int H, int D,
                     int stride, cudaStream_t s)
    {
        static bool attr_set = false;
        if (!attr_set)
        {
            cudaFuncSetAttribute(gated_delta_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 3 * D * sizeof(float) + (size_t)D * D * sizeof(float));
            attr_set = true;
        }
        gated_delta_kernel<<<H, D, 3 * D * sizeof(float) + (size_t)D * D * sizeof(float), s>>>(
            q, k, v, beta, g, state, out, seq, H, D, stride);
    }

    // ---------------- 全注意力 (GQA, 因果) ----------------

    __global__ void attn_kernel(const bf16* q, const bf16* kc, const bf16* vc, bf16* out,
                                int nq, int len, int Hq, int Hkv, int D, int base,
                                bool causal, float scale)
    {
        int h = blockIdx.y;
        int qrow = blockIdx.x;
        int groups = Hq / Hkv;
        int kvh = h / groups;
        int jmax = causal ? base + qrow : len - 1;
        if (jmax >= len) jmax = len - 1;
        const bf16* qr = q + ((size_t)qrow * Hq + h) * D;
        bf16* orow = out + ((size_t)qrow * Hq + h) * D;
        int t = threadIdx.x;
        extern __shared__ float sacc[];
        __shared__ float srd[256 / 8 + 1];
        for (int i = t; i < D; i += blockDim.x) sacc[i] = 0.f;
        __syncthreads();

        float m = -CUDART_INF_F, l = 0.f;
        for (int j = 0; j <= jmax; j++)
        {
            const bf16* kr = kc + ((size_t)j * Hkv + kvh) * D;
            float sd = 0.f;
            for (int i = t; i < D; i += blockDim.x) sd += b2f(qr[i]) * b2f(kr[i]);
            block_reduce(sd, srd, t, blockDim.x);
            sd *= scale;
            float mnew = fmaxf(m, sd);
            float p = __expf(sd - mnew);
            float corr = __expf(m - mnew);
            l = l * corr + p;
            const bf16* vr = vc + ((size_t)j * Hkv + kvh) * D;
            for (int i = t; i < D; i += blockDim.x) sacc[i] = sacc[i] * corr + p * b2f(vr[i]);
            m = mnew;
        }
        float inv = 1.f / l;
        for (int i = t; i < D; i += blockDim.x) orow[i] = f2b(sacc[i] * inv);
    }
    void attn_forward(const bf16* q, const bf16* kc, const bf16* vc, bf16* out,
                      int nq, int len, int Hq, int Hkv, int D, int base, bool causal,
                      float scale, cudaStream_t s)
    {
        static bool attr_set = false;
        if (!attr_set)
        {
            cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (size_t)D * sizeof(float));
            attr_set = true;
        }
        dim3 grid(nq, Hq);
        attn_kernel<<<grid, D, (size_t)D * sizeof(float), s>>>(q, kc, vc, out, nq, len, Hq,
                                                               Hkv, D, base, causal, scale);
    }

    // ---------------- 视觉注意力 (非因果, 分段) ----------------

    // ---------------- 视觉注意力 (非因果, 分段) ----------------
    // q/k/v: [seq, stride] 行主序交错 (q 列 0, k 列 D*H, v 列 2*D*H), out: [seq, H*D]

    __global__ void attn_vision_kernel(const bf16* q, const bf16* k, const bf16* v, bf16* out,
                                       int seq, int H, int D, const int* cu, int nseg,
                                       float scale, int stride)
    {
        int h = blockIdx.y;
        int qrow = blockIdx.x;
        int t = threadIdx.x;
        const bf16* qr = q + (size_t)qrow * stride + h * D;
        bf16* orow = out + ((size_t)qrow * H + h) * D;
        extern __shared__ float sacc[];
        __shared__ float srd[256 / 8 + 1];

        int seg = 0;
        while (seg + 1 < nseg && cu[seg + 1] <= qrow) seg++;
        int j0 = cu[seg], j1 = cu[seg + 1];

        for (int i = t; i < D; i += blockDim.x) sacc[i] = 0.f;
        __syncthreads();
        float m = -CUDART_INF_F, l = 0.f;
        for (int j = j0; j < j1; j++)
        {
            const bf16* kr = k + (size_t)j * stride + h * D;
            float sd = 0.f;
            for (int i = t; i < D; i += blockDim.x) sd += b2f(qr[i]) * b2f(kr[i]);
            block_reduce(sd, srd, t, blockDim.x);
            sd *= scale;
            float mnew = fmaxf(m, sd);
            float p = __expf(sd - mnew);
            float corr = __expf(m - mnew);
            l = l * corr + p;
            const bf16* vr = v + (size_t)j * stride + h * D;
            for (int i = t; i < D; i += blockDim.x) sacc[i] = sacc[i] * corr + p * b2f(vr[i]);
            m = mnew;
        }
        float inv = 1.f / l;
        for (int i = t; i < D; i += blockDim.x) orow[i] = f2b(sacc[i] * inv);
    }
    void attn_vision(const bf16* q, const bf16* k, const bf16* v, bf16* out,
                     int seq, int H, int D, const int* cu_seqlens, int nseg,
                     float scale, int stride, cudaStream_t s)
    {
        static bool attr_set = false;
        if (!attr_set)
        {
            cudaFuncSetAttribute(attn_vision_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (size_t)D * sizeof(float));
            attr_set = true;
        }
        dim3 grid(seq, H);
        attn_vision_kernel<<<grid, D, (size_t)D * sizeof(float), s>>>(
            q, k, v, out, seq, H, D, cu_seqlens, nseg, scale, stride);
    }

    // ---------------- 视觉 patch embed (Conv3d) ----------------

    __global__ void patch_embed_kernel(const float* img, const bf16* w, const bf16* bias,
                                       bf16* out, const int* order, int C, int T, int H,
                                       int W, int ph, int pw, int O)
    {
        int p = blockIdx.x;
        int j = threadIdx.x;
        int ph_idx = p / (W / pw);
        int pw_idx = p % (W / pw);
        __shared__ float swin[3 * 2 * 16 * 16];
        int win = C * T * ph * pw;
        // 权重布局 [O, C, T, ph, pw] 行主序: 线性索引 = ((cc*T+tt)*ph+pr)*pw+pc
        for (int i = j; i < win; i += blockDim.x)
        {
            int rem = i;
            int pc = rem % pw; rem /= pw;
            int pr = rem % ph; rem /= ph;
            int tt = rem % T; rem /= T;
            int cc = rem;
            int row = ph_idx * ph + pr;
            int col = pw_idx * pw + pc;
            swin[i] = img[((size_t)(cc * T + tt) * H + row) * W + col];
        }
        __syncthreads();
        const bf16* wj = w + j * win;
        float acc = b2f(bias[j]);
        for (int i = 0; i < win; i++) acc += b2f(wj[i]) * swin[i];
        out[(size_t)order[p] * O + j] = f2b(acc);
    }
    void patch_embed(const float* img, const bf16* w, const bf16* bias, bf16* out,
                     const int* order, int C, int T, int H, int W, int ph, int pw,
                     int O, cudaStream_t s)
    {
        int npatches = (H / ph) * (W / pw);
        patch_embed_kernel<<<npatches, O>>>(img, w, bias, out, order, C, T, H, W, ph, pw, O);
    }

    // ---------------- 位置嵌入双线性插值 ----------------

    __global__ void pos_embed_interp_kernel(const bf16* table, const int* idx,
                                            const float* wgt, bf16* out, int np, int D)
    {
        int p = blockIdx.x;
        int j = threadIdx.x;
        float acc = 0.f;
        for (int tap = 0; tap < 4; tap++)
            acc += wgt[p * 4 + tap] * b2f(table[(size_t)idx[p * 4 + tap] * D + j]);
        out[(size_t)p * D + j] = f2b(acc);
    }
    void pos_embed_interp(const bf16* table, const int* idx, const float* wgt,
                          bf16* out, int np, int D, cudaStream_t s)
    {
        pos_embed_interp_kernel<<<np, D>>>(table, idx, wgt, out, np, D);
    }

    // ---------------- LM head (bf16 输出 logits) ----------------

    __global__ void lm_head_kernel(const bf16* x, const bf16* embed, bf16* out, int V, int D)
    {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= V) return;
        const bf16* row = embed + (size_t)j * D;
        float acc = 0.f;
        for (int i = 0; i < D; i++) acc += b2f(row[i]) * b2f(x[i]);
        out[j] = f2b(acc);
    }
    void lm_head(const bf16* x, const bf16* embed, bf16* out, int V, int D, cudaStream_t s)
    {
        lm_head_kernel<<<(V + 255) / 256, 256, 0, s>>>(x, embed, out, V, D);
    }

    __global__ void argmax_bf16_kernel(const bf16* x, int* out, int n)
    {
        int t = threadIdx.x;
        float m = -CUDART_INF_F;
        int mi = 0;
        for (int i = t; i < n; i += blockDim.x)
        {
            float v = b2f(x[i]);
            if (v > m) { m = v; mi = i; }
        }
        // 先做 warp 归约 (否则同一 warp 各 lane 写同一 shared 槽, 结果任意)
        for (int off = 16; off > 0; off >>= 1)
        {
            float om = __shfl_down_sync(0xffffffffu, m, off);
            int omi = __shfl_down_sync(0xffffffffu, mi, off);
            if (om > m) { m = om; mi = omi; }
        }
        __shared__ float sm[512 / 32];
        __shared__ int si[512 / 32];
        if ((t & 31) == 0)
        {
            sm[t >> 5] = m;
            si[t >> 5] = mi;
        }
        __syncthreads();
        if (t == 0)
        {
            float gm = sm[0];
            int gi = si[0];
            for (int i = 1; i < blockDim.x / 32; i++)
                if (sm[i] > gm) { gm = sm[i]; gi = si[i]; }
            out[0] = gi;
        }
    }
    int argmax(const bf16* x, int n, cudaStream_t s)
    {
        static int* d_out = [] {
            int* p = nullptr;
            cudaMalloc(&p, 4);
            return p;
        }();
        argmax_bf16_kernel<<<1, 512, 0, s>>>(x, d_out, n);
        cudaStreamSynchronize(s);
        int r = 0;
        cudaMemcpy(&r, d_out, sizeof(int), cudaMemcpyDeviceToHost);
        return r;
    }
}
