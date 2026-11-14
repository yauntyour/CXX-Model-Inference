// ============================================================
// 推理快路径算子实现
// ============================================================

#include "fast_ops.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

using qk::bf16;

namespace
{
    __device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
    __device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }

    // ---------------- GEMV: 每个 warp 负责一行权重 ----------------
    // W 行连续; 32 lane 各读 2 个 bf16, 一轮覆盖 128 字节, 合并访存.
    template <int WARPS>
    __global__ void gemv_kernel(const bf16* __restrict__ x, const bf16* __restrict__ W,
                                bf16* __restrict__ y, int N, int K,
                                const bf16* __restrict__ add, int use16)
    {
        const int warp = blockIdx.x * WARPS + (threadIdx.x >> 5);
        const int lane = threadIdx.x & 31;
        if (warp >= N) return;

        const bf16* row = W + (size_t)warp * K;
        float acc = 0.f;

        // 16 字节向量读: 每个 lane 一次读 8 个 bf16, 32 lane 覆盖 512 字节/指令.
        // 权重行首 n*K*2 在 K 为 8 的倍数时天然 16 字节对齐.
        const bool aligned16 = use16 &&
            ((((uintptr_t)row | (uintptr_t)x) & 15) == 0) && (K % 8 == 0);
        if (aligned16)
        {
            int i = lane * 8;
            for (; i + 7 < K; i += 256)
            {
                uint4 xv = *reinterpret_cast<const uint4*>(x + i);
                uint4 wv = *reinterpret_cast<const uint4*>(row + i);
                const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
                const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
                #pragma unroll
                for (int k = 0; k < 4; k++)
                {
                    float2 xf = __bfloat1622float2(xb[k]);
                    float2 wf = __bfloat1622float2(wb[k]);
                    acc += xf.x * wf.x + xf.y * wf.y;
                }
            }
            for (int j = i; j < K; j++) acc += b2f(x[j]) * b2f(row[j]);
        }
        else
        {
            const bool aligned4 = (((uintptr_t)row | (uintptr_t)x) & 3) == 0;
            if (aligned4)
            {
                int i = lane * 2;
                for (; i + 1 < K; i += 64)
                {
                    __nv_bfloat162 xa = *reinterpret_cast<const __nv_bfloat162*>(x + i);
                    __nv_bfloat162 wa = *reinterpret_cast<const __nv_bfloat162*>(row + i);
                    acc += __bfloat162float(xa.x) * __bfloat162float(wa.x);
                    acc += __bfloat162float(xa.y) * __bfloat162float(wa.y);
                }
                if (i < K) acc += b2f(x[i]) * b2f(row[i]);
            }
            else
            {
                for (int i = lane; i < K; i += 32)
                    acc += b2f(x[i]) * b2f(row[i]);
            }
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);

        if (lane == 0)
        {
            if (add) acc += b2f(add[warp]);
            y[warp] = f2b(acc);
        }
    }

    // ---------------- W8A16 量化 ----------------
    __global__ void quantize_rows_kernel(const bf16* __restrict__ W,
                                         int8_t* __restrict__ Q,
                                         float* __restrict__ scale,
                                         int K)
    {
        const int row = blockIdx.x;
        const bf16* w = W + (size_t)row * K;
        int8_t* q = Q + (size_t)row * K;
        const int tid = threadIdx.x;

        float mx = 0.f;
        for (int i = tid; i < K; i += blockDim.x)
            mx = fmaxf(mx, fabsf(b2f(w[i])));

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));

        __shared__ float sm[32];
        if ((tid & 31) == 0) sm[tid >> 5] = mx;
        __syncthreads();
        if (tid == 0)
        {
            float m = sm[0];
            for (int i = 1; i < (int)(blockDim.x >> 5); i++) m = fmaxf(m, sm[i]);
            sm[0] = m;
        }
        __syncthreads();

        float s = sm[0] > 0.f ? sm[0] / 127.f : 1e-12f;
        float inv = 1.f / s;
        if (tid == 0) scale[row] = s;
        for (int i = tid; i < K; i += blockDim.x)
        {
            int v = __float2int_rn(b2f(w[i]) * inv);
            v = max(-127, min(127, v));
            q[i] = (int8_t)v;
        }
    }

    template <int WARPS>
    __global__ void gemv_w8a16_kernel(const bf16* __restrict__ x,
                                      const int8_t* __restrict__ Q,
                                      const float* __restrict__ scale,
                                      bf16* __restrict__ y, int N, int K,
                                      const bf16* __restrict__ add)
    {
        const int warp = blockIdx.x * WARPS + (threadIdx.x >> 5);
        const int lane = threadIdx.x & 31;
        if (warp >= N) return;

        const int8_t* row = Q + (size_t)warp * K;
        float acc = 0.f;

        const bool aligned = (K % 16 == 0) &&
            ((((uintptr_t)row | (uintptr_t)x) & 15) == 0);
        if (aligned)
        {
            int i = lane * 16;
            for (; i + 15 < K; i += 512)
            {
                uint4 wv = *reinterpret_cast<const uint4*>(row + i);
                uint4 xv0 = *reinterpret_cast<const uint4*>(x + i);
                uint4 xv1 = *reinterpret_cast<const uint4*>(x + i + 8);
                const int8_t* wb = reinterpret_cast<const int8_t*>(&wv);
                const __nv_bfloat162* xb0 =
                    reinterpret_cast<const __nv_bfloat162*>(&xv0);
                const __nv_bfloat162* xb1 =
                    reinterpret_cast<const __nv_bfloat162*>(&xv1);

                #pragma unroll
                for (int k = 0; k < 4; k++)
                {
                    float2 xf = __bfloat1622float2(xb0[k]);
                    acc += (float)wb[2 * k] * xf.x + (float)wb[2 * k + 1] * xf.y;
                }
                #pragma unroll
                for (int k = 0; k < 4; k++)
                {
                    float2 xf = __bfloat1622float2(xb1[k]);
                    acc += (float)wb[8 + 2 * k] * xf.x +
                           (float)wb[9 + 2 * k] * xf.y;
                }
            }
            for (int j = i; j < K; j++) acc += (float)row[j] * b2f(x[j]);
        }
        else
        {
            for (int j = lane; j < K; j += 32)
                acc += (float)row[j] * b2f(x[j]);
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);

        if (lane == 0)
        {
            float v = acc * scale[warp];
            if (add) v += b2f(add[warp]);
            y[warp] = f2b(v);
        }
    }

    cublasHandle_t get_handle()
    {
        static cublasHandle_t h = nullptr;
        if (!h)
        {
            cublasStatus_t st = cublasCreate(&h);
            if (st != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("cublasCreate failed");
        }
        return h;
    }

    // ---------------- 融合 QKV + RoPE + KV cache 写入 ----------------
    __global__ void qkv_rope_kv_kernel(const bf16* __restrict__ qkv,
                                       const int* __restrict__ pos,
                                       const float* __restrict__ inv_freq,
                                       bf16* __restrict__ q_out,
                                       bf16* __restrict__ kc,
                                       bf16* __restrict__ vc,
                                       int base, int S, int Hq, int Hkv, int D,
                                       int rot, int interleaved)
    {
        const int t = blockIdx.x;
        const int h = blockIdx.y;
        const int tid = threadIdx.x;
        const int half = rot / 2;
        const int Nq = Hq * D;
        const int Nk = Hkv * D;
        const bf16* row = qkv + (size_t)t * (Nq + 2 * Nk);
        const float p = (float)pos[t];

        if (h < Hq)
        {
            const bf16* qs = row + (size_t)h * D;
            bf16* qd = q_out + ((size_t)t * Hq + h) * D;
            for (int j = tid; j < half; j += blockDim.x)
            {
                float m = p * inv_freq[j];
                float c = __cosf(m), s = __sinf(m);
                int i0 = interleaved ? 2 * j : j;
                int i1 = interleaved ? 2 * j + 1 : j + half;
                float a = b2f(qs[i0]);
                float b = b2f(qs[i1]);
                qd[i0] = f2b(a * c - b * s);
                qd[i1] = f2b(a * s + b * c);
            }
            for (int d = rot + tid; d < D; d += blockDim.x) qd[d] = qs[d];
        }

        if (h < Hkv)
        {
            const bf16* ks = row + Nq + (size_t)h * D;
            bf16* kd = kc + ((size_t)(base + t) * Hkv + h) * D;
            for (int j = tid; j < half; j += blockDim.x)
            {
                float m = p * inv_freq[j];
                float c = __cosf(m), s = __sinf(m);
                int i0 = interleaved ? 2 * j : j;
                int i1 = interleaved ? 2 * j + 1 : j + half;
                float a = b2f(ks[i0]);
                float b = b2f(ks[i1]);
                kd[i0] = f2b(a * c - b * s);
                kd[i1] = f2b(a * s + b * c);
            }
            for (int d = rot + tid; d < D; d += blockDim.x) kd[d] = ks[d];

            const bf16* vs = row + Nq + Nk + (size_t)h * D;
            bf16* vd = vc + ((size_t)(base + t) * Hkv + h) * D;
            for (int d = tid; d < D; d += blockDim.x) vd[d] = vs[d];
        }
    }

    // ---------------- 融合 SwiGLU ----------------
    __global__ void swiglu_kernel(const bf16* __restrict__ gu, bf16* __restrict__ out,
                                  int rows, int I)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= rows * I) return;
        int r = idx / I;
        int i = idx - r * I;
        const bf16* row = gu + (size_t)r * 2 * I;
        float g = b2f(row[i]);
        float u = b2f(row[I + i]);
        float s = g / (1.f + __expf(-g));
        out[idx] = f2b(s * u);
    }

    // ---------------- decode 注意力: 一个 KV head 一个 block ----------------
    // ---------------- decode 注意力 split-K ----------------
    // 长上下文时把一个 KV head 的 key 区间切给多个 block, 提高 SM 占用;
    // 每个 block 输出局部 (m, l, o), 再由 combine kernel 合并.
    template <int ND>
    __global__ void decode_attn_part_kernel(const bf16* __restrict__ q,
                                            const bf16* __restrict__ kc,
                                            const bf16* __restrict__ vc,
                                            float* __restrict__ part_o,
                                            float* __restrict__ part_m,
                                            float* __restrict__ part_l,
                                            int len, int splits, int Hq, int Hkv,
                                            int D, float scale)
    {
        const int kvh = blockIdx.x;
        const int sp = blockIdx.y;
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int group = Hq / Hkv;
        if (warp >= group) return;

        const int h = kvh * group + warp;
        const int d0 = lane * ND;
        const int j0 = (int)((long long)len * sp / splits);
        const int j1 = (int)((long long)len * (sp + 1) / splits);

        float qreg[ND];
        float oreg[ND];
        #pragma unroll
        for (int k = 0; k < ND; k++)
        {
            qreg[k] = b2f(q[h * D + d0 + k]);
            oreg[k] = 0.f;
        }

        float m = -INFINITY;
        float l = 0.f;
        for (int j = j0; j < j1; j++)
        {
            const bf16* krow = kc + ((size_t)j * Hkv + kvh) * D;
            float dot = 0.f;
            #pragma unroll
            for (int k = 0; k < ND; k++) dot += qreg[k] * b2f(krow[d0 + k]);

            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_down_sync(0xffffffffu, dot, off);
            dot = __shfl_sync(0xffffffffu, dot, 0);

            const float sc = dot * scale;
            const float mn = fmaxf(m, sc);
            const float p = __expf(sc - mn);
            const float corr = __expf(m - mn);

            l = l * corr + p;
            const bf16* vrow = vc + ((size_t)j * Hkv + kvh) * D;
            #pragma unroll
            for (int k = 0; k < ND; k++)
                oreg[k] = oreg[k] * corr + p * b2f(vrow[d0 + k]);
            m = mn;
        }

        if (lane == 0)
        {
            part_m[h * splits + sp] = m;
            part_l[h * splits + sp] = l;
        }
        float* po = part_o + (size_t)(h * splits + sp) * D;
        #pragma unroll
        for (int k = 0; k < ND; k++) po[d0 + k] = oreg[k];
    }

    __global__ void decode_attn_combine_kernel(const float* __restrict__ part_o,
                                               const float* __restrict__ part_m,
                                               const float* __restrict__ part_l,
                                               bf16* __restrict__ out,
                                               int Hq, int splits, int D)
    {
        const int h = blockIdx.x;
        float M = -INFINITY;
        for (int s = 0; s < splits; s++) M = fmaxf(M, part_m[h * splits + s]);
        float L = 0.f;
        for (int s = 0; s < splits; s++)
            L += part_l[h * splits + s] * __expf(part_m[h * splits + s] - M);
        const float inv = 1.f / L;

        for (int d = threadIdx.x; d < D; d += blockDim.x)
        {
            float o = 0.f;
            for (int s = 0; s < splits; s++)
                o += part_o[(size_t)(h * splits + s) * D + d] *
                     __expf(part_m[h * splits + s] - M);
            out[h * D + d] = f2b(o * inv);
        }
    }

    float* g_part_ws = nullptr;
    size_t g_part_cap = 0;

    float* ensure_part_ws(size_t nfloats)
    {
        if (g_part_cap < nfloats)
        {
            if (g_part_ws) cudaFree(g_part_ws);
            g_part_ws = nullptr;
            cudaError_t e = cudaMalloc((void**)&g_part_ws, nfloats * sizeof(float));
            if (e != cudaSuccess)
                throw std::runtime_error("decode_attn split workspace alloc failed");
            g_part_cap = nfloats;
        }
        return g_part_ws;
    }

    template <int ND>
    void launch_decode_split(const bf16* q, const bf16* kc, const bf16* vc,
                             bf16* out, int len, int Hq, int Hkv, int D,
                             float scale, int splits, cudaStream_t s)
    {
        const size_t n_o = (size_t)Hq * splits * D;
        const size_t n_ml = (size_t)2 * Hq * splits;
        float* ws = ensure_part_ws(n_o + n_ml);
        float* po = ws;
        float* pm = ws + n_o;
        float* pl = pm + (size_t)Hq * splits;

        int group = Hq / Hkv;
        dim3 grid((unsigned)Hkv, (unsigned)splits);
        decode_attn_part_kernel<ND><<<grid, group * 32, 0, s>>>(
            q, kc, vc, po, pm, pl, len, splits, Hq, Hkv, D, scale);

        int threads = D < 256 ? D : 256;
        decode_attn_combine_kernel<<<Hq, threads, 0, s>>>(po, pm, pl, out, Hq, splits, D);
    }

    // ---------------- prefill 闪存注意力 (bf16 WMMA, 两遍, D=128) ----------------
    // 一个 block = 16 行 query × 1 个 head; K/V tile = 64 行.
    // 第一遍求每行 lse, 第二遍重算 S 并累加 O (避免 online rescale 的 smem 往返).
    constexpr int FPQ = 16;
    constexpr int FPK = 64;
    constexpr int FPLD = 144;   // 144*2=288 字节, 32 字节对齐
    constexpr int FPLDP = 80;   // P 的 smem 行跨距

    struct FlashSp
    {
        float s[FPQ * FPK];
        bf16 p[FPQ * FPLDP];
    };
    union FlashScratch
    {
        FlashSp sp;
        float o[FPQ * 128];   // 循环结束后复用 s/p 的空间
    };
    struct FlashSmem
    {
        bf16 q[FPQ * FPLD];
        bf16 k[FPK * FPLD];
        bf16 v[FPK * FPLD];
        float m[FPQ];
        float l[FPQ];
        FlashScratch scratch;
    };

    __global__ void flash_prefill_kernel(const bf16* __restrict__ q,
                                         const bf16* __restrict__ kc,
                                         const bf16* __restrict__ vc,
                                         bf16* __restrict__ out,
                                         int base, int S, int len, int Hq, int Hkv,
                                         int D, float scale)
    {
        using namespace nvcuda::wmma;
        extern __shared__ char smem_raw[];
        FlashSmem& sm = *reinterpret_cast<FlashSmem*>(smem_raw);

        const int q0 = blockIdx.x * FPQ;
        const int h = blockIdx.y;
        const int tid = threadIdx.x;
        const int warp = tid >> 5;
        const int group = Hq / Hkv;
        const int kvh = h / group;
        const int rows = min(FPQ, S - q0);
        if (rows <= 0) return;

        // Q tile
        for (int idx = tid; idx < FPQ * D; idx += blockDim.x)
        {
            int r = idx / D, d = idx - r * D;
            sm.q[r * FPLD + d] =
                (r < rows) ? q[((size_t)(q0 + r) * Hq + h) * D + d] : f2b(0.f);
        }
        if (tid < FPQ)
        {
            sm.m[tid] = -INFINITY;
            sm.l[tid] = 0.f;
        }

        // 增量 prefill 时 q 段从 base 开始; key 仍从 0 扫到 len.
        const int qbase = base + q0;
        const int kmax = min(len, qbase + rows);   // causal 上界 (不含)
        const int nkt = (kmax + FPK - 1) / FPK;
        const float sc = scale;

        // ---- pass 1: 行 lse ----
        for (int kt = 0; kt < nkt; kt++)
        {
            const int k0 = kt * FPK;
            const int krows = min(FPK, len - k0);

            for (int idx = tid; idx < FPK * D; idx += blockDim.x)
            {
                int r = idx / D, d = idx - r * D;
                int j = k0 + r;
                sm.k[r * FPLD + d] =
                    (r < krows) ? kc[((size_t)j * Hkv + kvh) * D + d] : f2b(0.f);
            }
            __syncthreads();

            {
                fragment<accumulator, 16, 16, 16, float> acc;
                fill_fragment(acc, 0.f);
                fragment<matrix_a, 16, 16, 16, bf16, row_major> af;
                fragment<matrix_b, 16, 16, 16, bf16, col_major> bf;
                #pragma unroll
                for (int d = 0; d < 128; d += 16)
                {
                    load_matrix_sync(af, sm.q + d, FPLD);
                    load_matrix_sync(bf, sm.k + (warp * 16) * FPLD + d, FPLD);
                    mma_sync(acc, af, bf, acc);
                }
                store_matrix_sync(sm.scratch.sp.s + warp * 16, acc, FPK, mem_row_major);
            }
            __syncthreads();

            if (tid < FPQ)
            {
                const int r = tid;
                const int jmax = min(kmax, qbase + r + 1);
                float mx = -INFINITY;
                for (int c = 0; c < FPK; c++)
                {
                    int j = k0 + c;
                    if (j >= jmax) break;
                    mx = fmaxf(mx, sm.scratch.sp.s[r * FPK + c] * sc);
                }
                if (mx > -INFINITY)
                {
                    float sum = 0.f;
                    for (int c = 0; c < FPK; c++)
                    {
                        int j = k0 + c;
                        if (j >= jmax) break;
                        sum += __expf(sm.scratch.sp.s[r * FPK + c] * sc - mx);
                    }
                    float mn = fmaxf(sm.m[r], mx);
                    sm.l[r] = sm.l[r] * __expf(sm.m[r] - mn) + sum * __expf(mx - mn);
                    sm.m[r] = mn;
                }
            }
            __syncthreads();
        }

        if (tid < FPQ)
            sm.m[tid] = (sm.l[tid] > 0.f) ? (sm.m[tid] + __logf(sm.l[tid])) : 0.f;

        // ---- pass 2: O = P @ V ----
        fragment<accumulator, 16, 16, 16, float> oacc[2];
        fill_fragment(oacc[0], 0.f);
        fill_fragment(oacc[1], 0.f);

        for (int kt = 0; kt < nkt; kt++)
        {
            const int k0 = kt * FPK;
            const int krows = min(FPK, len - k0);

            for (int idx = tid; idx < FPK * D; idx += blockDim.x)
            {
                int r = idx / D, d = idx - r * D;
                int j = k0 + r;
                bool ok = r < krows;
                sm.k[r * FPLD + d] =
                    ok ? kc[((size_t)j * Hkv + kvh) * D + d] : f2b(0.f);
                sm.v[r * FPLD + d] =
                    ok ? vc[((size_t)j * Hkv + kvh) * D + d] : f2b(0.f);
            }
            __syncthreads();

            {
                fragment<accumulator, 16, 16, 16, float> acc;
                fill_fragment(acc, 0.f);
                fragment<matrix_a, 16, 16, 16, bf16, row_major> af;
                fragment<matrix_b, 16, 16, 16, bf16, col_major> bf;
                #pragma unroll
                for (int d = 0; d < 128; d += 16)
                {
                    load_matrix_sync(af, sm.q + d, FPLD);
                    load_matrix_sync(bf, sm.k + (warp * 16) * FPLD + d, FPLD);
                    mma_sync(acc, af, bf, acc);
                }
                store_matrix_sync(sm.scratch.sp.s + warp * 16, acc, FPK, mem_row_major);
            }
            __syncthreads();

            for (int idx = tid; idx < FPQ * FPK; idx += blockDim.x)
            {
                int r = idx / FPK, c = idx - r * FPK;
                int j = k0 + c;
                float pv = 0.f;
                if (r < rows && c < krows && j <= qbase + r)
                    pv = __expf(sm.scratch.sp.s[r * FPK + c] * sc - sm.m[r]);
                sm.scratch.sp.p[r * FPLDP + c] = f2b(pv);
            }
            __syncthreads();

            #pragma unroll
            for (int n = 0; n < 2; n++)
            {
                int col = warp * 32 + n * 16;
                fragment<matrix_a, 16, 16, 16, bf16, row_major> af;
                fragment<matrix_b, 16, 16, 16, bf16, row_major> bf;
                #pragma unroll
                for (int d = 0; d < FPK; d += 16)
                {
                    load_matrix_sync(af, sm.scratch.sp.p + d, FPLDP);
                    load_matrix_sync(bf, sm.v + (size_t)d * FPLD + col, FPLD);
                    mma_sync(oacc[n], af, bf, oacc[n]);
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int n = 0; n < 2; n++)
        {
            int col = warp * 32 + n * 16;
            store_matrix_sync(sm.scratch.o + col, oacc[n], 128, mem_row_major);
        }
        __syncthreads();

        for (int idx = tid; idx < rows * D; idx += blockDim.x)
        {
            int r = idx / D, d = idx - r * D;
            out[((size_t)(q0 + r) * Hq + h) * D + d] = f2b(sm.scratch.o[r * 128 + d]);
        }
    }

    // ---------------- prefill 闪存注意力 32-row 版本 ----------------
    // 32 行 query × 64 行 key, 8 warps; K/V tile 每 32 个 query 行只读一次.
    constexpr int F32Q = 32;
    constexpr int F32K = 64;
    constexpr int F32LD = 144;
    constexpr int F32LDP = 80;

    struct Flash32Sp
    {
        float s[F32Q * F32K];
        bf16 p[F32Q * F32LDP];
    };
    union Flash32Scratch
    {
        Flash32Sp sp;
        float o[F32Q * 128];
    };
    struct Flash32Smem
    {
        bf16 q[F32Q * F32LD];
        bf16 k[F32K * F32LD];
        bf16 v[F32K * F32LD];
        float m[F32Q];
        float l[F32Q];
        Flash32Scratch scratch;
    };

    __global__ void flash_prefill32_kernel(const bf16* __restrict__ q,
                                           const bf16* __restrict__ kc,
                                           const bf16* __restrict__ vc,
                                           bf16* __restrict__ out,
                                           int base, int S, int len, int Hq, int Hkv,
                                           int D, float scale)
    {
        using namespace nvcuda::wmma;
        extern __shared__ char smem_raw[];
        Flash32Smem& sm = *reinterpret_cast<Flash32Smem*>(smem_raw);

        const int q0 = blockIdx.x * F32Q;
        const int h = blockIdx.y;
        const int tid = threadIdx.x;
        const int warp = tid >> 5;
        const int group = Hq / Hkv;
        const int kvh = h / group;
        const int rows = min(F32Q, S - q0);
        if (rows <= 0) return;

        for (int idx = tid; idx < F32Q * D; idx += blockDim.x)
        {
            int r = idx / D, d = idx - r * D;
            sm.q[r * F32LD + d] =
                (r < rows) ? q[((size_t)(q0 + r) * Hq + h) * D + d] : f2b(0.f);
        }
        if (tid < F32Q)
        {
            sm.m[tid] = -INFINITY;
            sm.l[tid] = 0.f;
        }

        const int qbase = base + q0;
        const int kmax = min(len, qbase + rows);
        const int nkt = (kmax + F32K - 1) / F32K;
        const float sc = scale;

        // pass 1: lse
        for (int kt = 0; kt < nkt; kt++)
        {
            const int k0 = kt * F32K;
            const int krows = min(F32K, len - k0);

            for (int idx = tid; idx < F32K * D; idx += blockDim.x)
            {
                int r = idx / D, d = idx - r * D;
                int j = k0 + r;
                sm.k[r * F32LD + d] =
                    (r < krows) ? kc[((size_t)j * Hkv + kvh) * D + d] : f2b(0.f);
            }
            __syncthreads();

            {
                const int qh = warp >> 2;
                const int kq = warp & 3;
                fragment<accumulator, 16, 16, 16, float> acc;
                fill_fragment(acc, 0.f);
                fragment<matrix_a, 16, 16, 16, bf16, row_major> af;
                fragment<matrix_b, 16, 16, 16, bf16, col_major> bf;
                #pragma unroll
                for (int d = 0; d < 128; d += 16)
                {
                    load_matrix_sync(af, sm.q + (size_t)(qh * 16) * F32LD + d, F32LD);
                    load_matrix_sync(bf, sm.k + (size_t)(kq * 16) * F32LD + d, F32LD);
                    mma_sync(acc, af, bf, acc);
                }
                store_matrix_sync(sm.scratch.sp.s + (size_t)(qh * 16) * F32K + kq * 16,
                                  acc, F32K, mem_row_major);
            }
            __syncthreads();

            if (tid < F32Q)
            {
                const int r = tid;
                const int jmax = min(kmax, qbase + r + 1);
                float mx = -INFINITY;
                for (int c = 0; c < F32K; c++)
                {
                    int j = k0 + c;
                    if (j >= jmax) break;
                    mx = fmaxf(mx, sm.scratch.sp.s[r * F32K + c] * sc);
                }
                if (mx > -INFINITY)
                {
                    float sum = 0.f;
                    for (int c = 0; c < F32K; c++)
                    {
                        int j = k0 + c;
                        if (j >= jmax) break;
                        sum += __expf(sm.scratch.sp.s[r * F32K + c] * sc - mx);
                    }
                    float mn = fmaxf(sm.m[r], mx);
                    sm.l[r] = sm.l[r] * __expf(sm.m[r] - mn) + sum * __expf(mx - mn);
                    sm.m[r] = mn;
                }
            }
            __syncthreads();
        }

        if (tid < F32Q)
            sm.m[tid] = (sm.l[tid] > 0.f) ? (sm.m[tid] + __logf(sm.l[tid])) : 0.f;

        // pass 2: O = P @ V
        fragment<accumulator, 16, 16, 16, float> oacc[2];
        fill_fragment(oacc[0], 0.f);
        fill_fragment(oacc[1], 0.f);

        for (int kt = 0; kt < nkt; kt++)
        {
            const int k0 = kt * F32K;
            const int krows = min(F32K, len - k0);

            for (int idx = tid; idx < F32K * D; idx += blockDim.x)
            {
                int r = idx / D, d = idx - r * D;
                int j = k0 + r;
                bool ok = r < krows;
                sm.k[r * F32LD + d] =
                    ok ? kc[((size_t)j * Hkv + kvh) * D + d] : f2b(0.f);
                sm.v[r * F32LD + d] =
                    ok ? vc[((size_t)j * Hkv + kvh) * D + d] : f2b(0.f);
            }
            __syncthreads();

            {
                const int qh = warp >> 2;
                const int kq = warp & 3;
                fragment<accumulator, 16, 16, 16, float> acc;
                fill_fragment(acc, 0.f);
                fragment<matrix_a, 16, 16, 16, bf16, row_major> af;
                fragment<matrix_b, 16, 16, 16, bf16, col_major> bf;
                #pragma unroll
                for (int d = 0; d < 128; d += 16)
                {
                    load_matrix_sync(af, sm.q + (size_t)(qh * 16) * F32LD + d, F32LD);
                    load_matrix_sync(bf, sm.k + (size_t)(kq * 16) * F32LD + d, F32LD);
                    mma_sync(acc, af, bf, acc);
                }
                store_matrix_sync(sm.scratch.sp.s + (size_t)(qh * 16) * F32K + kq * 16,
                                  acc, F32K, mem_row_major);
            }
            __syncthreads();

            for (int idx = tid; idx < F32Q * F32K; idx += blockDim.x)
            {
                int r = idx / F32K, c = idx - r * F32K;
                int j = k0 + c;
                float pv = 0.f;
                if (r < rows && c < krows && j <= qbase + r)
                    pv = __expf(sm.scratch.sp.s[r * F32K + c] * sc - sm.m[r]);
                sm.scratch.sp.p[r * F32LDP + c] = f2b(pv);
            }
            __syncthreads();

            {
                const int qh = warp >> 2;
                const int sub = warp & 3;
                #pragma unroll
                for (int n = 0; n < 2; n++)
                {
                    int col = sub * 32 + n * 16;
                    fragment<matrix_a, 16, 16, 16, bf16, row_major> af;
                    fragment<matrix_b, 16, 16, 16, bf16, row_major> bf;
                    #pragma unroll
                    for (int d = 0; d < F32K; d += 16)
                    {
                        load_matrix_sync(af, sm.scratch.sp.p +
                                             (size_t)(qh * 16) * F32LDP + d, F32LDP);
                        load_matrix_sync(bf, sm.v + (size_t)d * F32LD + col, F32LD);
                        mma_sync(oacc[n], af, bf, oacc[n]);
                    }
                }
            }
            __syncthreads();
        }

        {
            const int qh = warp >> 2;
            const int sub = warp & 3;
            #pragma unroll
            for (int n = 0; n < 2; n++)
            {
                int col = sub * 32 + n * 16;
                store_matrix_sync(sm.scratch.o + (size_t)(qh * 16) * 128 + col,
                                  oacc[n], 128, mem_row_major);
            }
        }
        __syncthreads();

        for (int idx = tid; idx < rows * D; idx += blockDim.x)
        {
            int r = idx / D, d = idx - r * D;
            out[((size_t)(q0 + r) * Hq + h) * D + d] = f2b(sm.scratch.o[r * 128 + d]);
        }
    }

    template <int ND>
    __global__ void decode_attn_kernel(const bf16* __restrict__ q,
                                       const bf16* __restrict__ kc,
                                       const bf16* __restrict__ vc,
                                       bf16* __restrict__ out,
                                       int len, int Hq, int Hkv, int D, float scale)
    {
        const int kvh = blockIdx.x;
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int group = Hq / Hkv;
        if (warp >= group) return;

        const int h = kvh * group + warp;
        const int d0 = lane * ND;

        float qreg[ND];
        float oreg[ND];
        #pragma unroll
        for (int k = 0; k < ND; k++)
        {
            qreg[k] = b2f(q[h * D + d0 + k]);
            oreg[k] = 0.f;
        }

        float m = -INFINITY;
        float l = 0.f;
        for (int j = 0; j < len; j++)
        {
            const bf16* krow = kc + ((size_t)j * Hkv + kvh) * D;
            float dot = 0.f;
            #pragma unroll
            for (int k = 0; k < ND; k++) dot += qreg[k] * b2f(krow[d0 + k]);

            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_down_sync(0xffffffffu, dot, off);
            dot = __shfl_sync(0xffffffffu, dot, 0);

            const float sc = dot * scale;
            const float mn = fmaxf(m, sc);
            const float p = __expf(sc - mn);
            const float corr = __expf(m - mn);

            l = l * corr + p;
            const bf16* vrow = vc + ((size_t)j * Hkv + kvh) * D;
            #pragma unroll
            for (int k = 0; k < ND; k++)
                oreg[k] = oreg[k] * corr + p * b2f(vrow[d0 + k]);
            m = mn;
        }

        const float inv = 1.f / l;
        #pragma unroll
        for (int k = 0; k < ND; k++)
            out[h * D + d0 + k] = f2b(oreg[k] * inv);
    }
} // namespace

namespace fo
{
    void gemv_bf16(const bf16* x, const bf16* W, bf16* y,
                   int N, int K, const bf16* add, cudaStream_t s)
    {
        if (N <= 0 || K <= 0) return;
        static const bool force4 = std::getenv("FO_GEMV4") != nullptr;
        constexpr int WARPS = 8;
        int blocks = (N + WARPS - 1) / WARPS;
        gemv_kernel<WARPS><<<blocks, WARPS * 32, 0, s>>>(x, W, y, N, K, add,
                                                         force4 ? 0 : 1);
    }

    void quantize_w8a16(const bf16* W, int8_t* Q, float* scale,
                        int N, int K, cudaStream_t s)
    {
        if (N <= 0 || K <= 0) return;
        quantize_rows_kernel<<<N, 256, 0, s>>>(W, Q, scale, K);
    }

    void gemv_w8a16(const bf16* x, const int8_t* Q, const float* scale,
                    bf16* y, int N, int K, const bf16* add, cudaStream_t s)
    {
        if (N <= 0 || K <= 0) return;
        constexpr int WARPS = 8;
        int blocks = (N + WARPS - 1) / WARPS;
        gemv_w8a16_kernel<WARPS><<<blocks, WARPS * 32, 0, s>>>(
            x, Q, scale, y, N, K, add);
    }

    void gemm_nt_bf16(const bf16* A, const bf16* W, bf16* C,
                      int M, int N, int K, float beta, cudaStream_t s)
    {
        if (M <= 0 || N <= 0 || K <= 0) return;

        if (M == 1)
        {
            // decode: 自写 GEMV; beta=1 时 C 同时作为残差输入
            gemv_bf16(A, W, C, N, K, beta != 0.f ? C : nullptr, s);
            return;
        }

        float alpha = 1.0f;
        cublasHandle_t h = get_handle();
        cublasSetStream(h, s);
        cublasStatus_t st = cublasGemmEx(
            h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
            W, CUDA_R_16BF, K,
            A, CUDA_R_16BF, K,
            &beta, C, CUDA_R_16BF, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasGemmEx(NT, bf16) failed: " +
                                     std::to_string((int)st));
    }

    void qkv_rope_kv(const bf16* qkv, const int* pos, const float* inv_freq,
                     bf16* q_out, bf16* kc, bf16* vc,
                     int base, int S, int Hq, int Hkv, int D, int rot,
                     int interleaved, cudaStream_t s)
    {
        if (S <= 0) return;
        dim3 grid((unsigned)S, (unsigned)(Hq > Hkv ? Hq : Hkv));
        qkv_rope_kv_kernel<<<grid, 128, 0, s>>>(qkv, pos, inv_freq, q_out, kc, vc,
                                                base, S, Hq, Hkv, D, rot, interleaved);
    }

    void swiglu_bf16(const bf16* gu, bf16* out, int rows, int I, cudaStream_t s)
    {
        int n = rows * I;
        if (n <= 0) return;
        swiglu_kernel<<<(n + 255) / 256, 256, 0, s>>>(gu, out, rows, I);
    }

    bool decode_attn_supported(int D)
    {
        return D == 64 || D == 128 || D == 256;
    }

    bool flash_attn_supported(int D)
    {
        return D == 128;
    }

    void flash_attn_prefill(const bf16* q, const bf16* kc, const bf16* vc,
                            bf16* out, int base, int S, int len, int Hq, int Hkv,
                            int D, float scale, cudaStream_t s)
    {
        if (S <= 0) return;
        if (D != 128)
            throw std::runtime_error("flash_attn_prefill: only D=128 is supported");

        // 长序列用 32-row 版本: K/V tile 每 32 个 query 行只读一次
        if (S >= 256)
        {
            static bool attr32 = false;
            if (!attr32)
            {
                cudaError_t e = cudaFuncSetAttribute(
                    flash_prefill32_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)sizeof(Flash32Smem));
                if (e != cudaSuccess)
                    throw std::runtime_error(std::string("flash32 smem attribute: ") +
                                             cudaGetErrorString(e));
                attr32 = true;
            }
            dim3 grid((unsigned)((S + F32Q - 1) / F32Q), (unsigned)Hq);
            flash_prefill32_kernel<<<grid, 256, sizeof(Flash32Smem), s>>>(
                q, kc, vc, out, base, S, len, Hq, Hkv, D, scale);
            return;
        }

        static bool attr_set = false;
        if (!attr_set)
        {
            cudaError_t e = cudaFuncSetAttribute(
                flash_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)sizeof(FlashSmem));
            if (e != cudaSuccess)
                throw std::runtime_error(std::string("flash smem attribute: ") +
                                         cudaGetErrorString(e));
            attr_set = true;
        }

        dim3 grid((unsigned)((S + FPQ - 1) / FPQ), (unsigned)Hq);
        flash_prefill_kernel<<<grid, 128, sizeof(FlashSmem), s>>>(
            q, kc, vc, out, base, S, len, Hq, Hkv, D, scale);
    }

    void decode_attn(const bf16* q, const bf16* kc, const bf16* vc,
                     bf16* out, int len, int Hq, int Hkv, int D,
                     float scale, cudaStream_t s)
    {
        // 长上下文: key 区间切给多个 block, 提高占用率
        static const bool no_splitk = std::getenv("FO_NO_SPLITK") != nullptr;
        int splits = 1;
        if (!no_splitk && len > 128)
            splits = std::min(32, (len + 127) / 128);
        if (splits > 1)
        {
            if (D == 64)
                launch_decode_split<2>(q, kc, vc, out, len, Hq, Hkv, D, scale, splits, s);
            else if (D == 128)
                launch_decode_split<4>(q, kc, vc, out, len, Hq, Hkv, D, scale, splits, s);
            else if (D == 256)
                launch_decode_split<8>(q, kc, vc, out, len, Hq, Hkv, D, scale, splits, s);
            else
                throw std::runtime_error("decode_attn: unsupported head_dim");
            return;
        }

        int group = Hq / Hkv;
        int threads = group * 32;
        if (D == 64)
            decode_attn_kernel<2><<<Hkv, threads, 0, s>>>(q, kc, vc, out, len, Hq, Hkv, D, scale);
        else if (D == 128)
            decode_attn_kernel<4><<<Hkv, threads, 0, s>>>(q, kc, vc, out, len, Hq, Hkv, D, scale);
        else if (D == 256)
            decode_attn_kernel<8><<<Hkv, threads, 0, s>>>(q, kc, vc, out, len, Hq, Hkv, D, scale);
        else
            throw std::runtime_error("decode_attn: unsupported head_dim");
    }
} // namespace fo
