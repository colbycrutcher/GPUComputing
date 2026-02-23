#include <stdio.h>
#include "kernel1.h"


extern  __shared__  float sdata[];

////////////////////////////////////////////////////////////////////////////////
//! Weighted Jacobi Iteration
//! @param g_dataA  input data in global memory
//! @param g_dataB  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void k1( float* g_dataA, float* g_dataB, int floatpitch, int width) 
{
    // Local aliases (don’t change parameters, but help intent/optimization)
    const float* __restrict__ gA = g_dataA;
    float*       __restrict__ gB = g_dataB;

    const int Bx = blockDim.x, By = blockDim.y;
    const int tileW = Bx + 2;
    const int tileH = By + 2;

    const int tx = threadIdx.x, ty = threadIdx.y;

    // Tile origin in global memory (tile includes halo)
    const int g0i = blockIdx.y * By;
    const int g0j = blockIdx.x * Bx;

    // Cooperative load: entire (By+2) x (Bx+2) tile into shared
    for (int y = ty; y < tileH; y += By) {
        const int gi = g0i + y;
        for (int x = tx; x < tileW; x += Bx) {
            const int gj = g0j + x;
            sdata[y * tileW + x] = (gi < width && gj < width) ? gA[gi * floatpitch + gj] : 0.0f;
        }
    }
    __syncthreads();

    // This thread computes one interior cell (matches k0’s +1 shift)
    const int i  = g0i + ty + 1;
    const int j  = g0j + tx + 1;
    const int si = ty + 1;
    const int sj = tx + 1;

    if (i < width - 1 && j < width - 1) {
        // NOTE: requires device lambda support in your NVCC flags.
        auto S = [&](int y, int x) -> float { return sdata[y * tileW + x]; };

        const float center = S(si, sj);
        const float sum8 =
            S(si-1, sj  ) + S(si-1, sj+1) + S(si,   sj+1) + S(si+1, sj+1) +
            S(si+1, sj  ) + S(si+1, sj-1) + S(si,   sj-1) + S(si-1, sj-1);

        gB[i * floatpitch + j] = 0.95f * (0.2f * center + 0.1f * sum8);
    }
}

