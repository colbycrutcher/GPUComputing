#include <cuda_runtime.h>
#include <math.h>
#include "pgmProcess.h"

__device__ float distance(int p1[], int p2[])
{
    int dr = p1[0] - p2[0];
    int dc = p1[1] - p2[1];
    return sqrtf((float)(dr * dr + dc * dc));
}

__global__ void kDrawCircle(int *pixels, int numRows, int numCols,
                            int centerRow, int centerCol, int radius)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= numRows || col >= numCols) return;

    int p[2]  = { row, col };
    int c[2]  = { centerRow, centerCol };

    if (distance(p, c) <= (float)radius) {
        pixels[row * numCols + col] = 0;
    }
}

__global__ void kDrawEdge(int *pixels, int numRows, int numCols, int edgeWidth)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= numRows || col >= numCols) return;

    if (row < edgeWidth || row >= (numRows - edgeWidth) ||
        col < edgeWidth || col >= (numCols - edgeWidth)) {
        pixels[row * numCols + col] = 0;
    }
}

__global__ void kDrawLine(int *pixels, int numRows, int numCols,
                          int p1row, int p1col, int p2row, int p2col)
{
    // draw by stepping t in [0,1]
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int dr = p2row - p1row;
    int dc = p2col - p1col;

    int steps = abs(dr);
    if (abs(dc) > steps) steps = abs(dc);
    if (steps == 0) steps = 1;

    if (tid > steps) return;

    float t = (float)tid / (float)steps;
    int r = (int)lrintf(p1row + t * dr);
    int c = (int)lrintf(p1col + t * dc);

    if (r >= 0 && r < numRows && c >= 0 && c < numCols) {
        pixels[r * numCols + c] = 0;
    }
}

static dim3 defaultBlock2D() { return dim3(16, 16, 1); }

static dim3 defaultGrid2D(int numCols, int numRows, dim3 block)
{
    return dim3((numCols + block.x - 1) / block.x,
                (numRows + block.y - 1) / block.y,
                1);
}

void launchDrawCircle(int *d_pixels, int numRows, int numCols,
                      int centerRow, int centerCol, int radius)
{
    dim3 block = defaultBlock2D();
    dim3 grid  = defaultGrid2D(numCols, numRows, block);
    kDrawCircle<<<grid, block>>>(d_pixels, numRows, numCols, centerRow, centerCol, radius);
}

void launchDrawEdge(int *d_pixels, int numRows, int numCols, int edgeWidth)
{
    dim3 block = defaultBlock2D();
    dim3 grid  = defaultGrid2D(numCols, numRows, block);
    kDrawEdge<<<grid, block>>>(d_pixels, numRows, numCols, edgeWidth);
}

void launchDrawLine(int *d_pixels, int numRows, int numCols,
                    int p1row, int p1col, int p2row, int p2col)
{
    // 1D launch (steps-based)
    int dr = p2row - p1row;
    int dc = p2col - p1col;
    int steps = abs(dr);
    if (abs(dc) > steps) steps = abs(dc);
    if (steps == 0) steps = 1;

    int threads = 256;
    int blocks = (steps + threads) / threads; // +1 safety baked in
    kDrawLine<<<blocks, threads>>>(d_pixels, numRows, numCols, p1row, p1col, p2row, p2col);
}
