#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#define TILE_SIZE 32

template <typename T>
__global__ void standard_attention_kernel(
    const T *q,
    const T *k,
    const T *v,
    const bool *mask,
    T *out,
    int B, int T, int Tp, int C, int Cp)
{
    __shared__ T q_shared[TILE_SIZE][TILE_SIZE];
    __shared__ T k_shared[TILE_SIZE][TILE_SIZE];
    __shared__ T v_shared[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int b = bx;
    int t = by * TILE_SIZE + ty;
    int c_ = bx * TILE_SIZE + tx;

    T scale = powf(C, -0.25f);
    T attn_sum = 0.0f;

    if (t < T && c_ < Cp)
    {
        for (int m = 0; m < (Tp - 1) / TILE_SIZE + 1; ++m)
        {
            if (m * TILE_SIZE + tx < Tp && t < T)
            {
                q_shared[ty][tx] = q[(b * T + t) * C + m * TILE_SIZE + tx] * scale;
                k_shared[ty][tx] = k[(b * Tp + m * TILE_SIZE + tx) * C + t] * scale;
            }
            else
            {
                q_shared[ty][tx] = 0;
                k_shared[ty][tx] = 0;
            }

            __syncthreads();

            for (int t_ = 0; t_ < TILE_SIZE; ++t_)
            {
                T score = q_shared[ty][t_] * k_shared[t_][tx];

                if (mask != nullptr && !mask[b * T * Tp + t * Tp + m * TILE_SIZE + t_])
                {
                    score = -INFINITY;
                }

                attn_sum += expf(score);
            }

            __syncthreads();
        }

        T out_value = 0.0f;

        for (int m = 0; m < (Tp - 1) / TILE_SIZE + 1; ++m)
        {
            if (m * TILE_SIZE + tx < Tp && t < T)
            {
                q_shared[ty][tx] = q[(b * T + t) * C + m * TILE_SIZE + tx] * scale;
                k_shared[ty][tx] = k[(b * Tp + m * TILE_SIZE + tx) * C + t] * scale;
            }
            else
            {
                q_shared[ty][tx] = 0;
                k_shared[ty][tx] = 0;
            }

            if (m * TILE_SIZE + tx < Tp && c_ < Cp)
            {
                v_shared[ty][tx] = v[(b * Tp + m * TILE_SIZE + tx) * Cp + c_];
            }
            else
            {
                v_shared[ty][tx] = 0;
            }

            __syncthreads();

            for (int t_ = 0; t_ < TILE_SIZE; ++t_)
            {
                T score = q_shared[ty][t_] * k_shared[t_][tx];

                if (mask != nullptr && !mask[b * T * Tp + t * Tp + m * TILE_SIZE + t_])
                {
                    score = -INFINITY;
                }

                T attn = expf(score) / attn_sum;

                out_value += attn * v_shared[t_][tx];
            }

            __syncthreads();
        }

        out[(b * T + t) * Cp + c_] = out_value;
    }
}
