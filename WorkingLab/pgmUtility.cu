#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "pgmUtility.h"
#include "pgmProcess.h"

// helper: recompute max and update header[3] if needed
static int updateHeaderMax(char **header, const int *pixels, int n)
{
    if (!header || !header[3] || !pixels || n <= 0) return 0;

    int oldMax = atoi(header[3]);
    int newMax = 0;
    for (int i = 0; i < n; i++) {
        if (pixels[i] > newMax) newMax = pixels[i];
    }

    if (newMax != oldMax) {
        snprintf(header[3], maxSizeHeadRow, "%d\n", newMax);
        return 1;
    }
    return 0;
}

int *pgmRead(char **header, int *numRows, int *numCols, FILE *in)
{
    if (!header || !numRows || !numCols || !in) return NULL;

    // allocate header rows and read them
    for (int i = 0; i < rowsInHeader; i++) {
        header[i] = (char*)calloc(maxSizeHeadRow, sizeof(char));
        if (!header[i]) return NULL;
        if (!fgets(header[i], maxSizeHeadRow, in)) return NULL;
    }

    // parse rows/cols from header[2] (typical format: "rows cols" OR "cols rows")
    // PGM standard is "cols rows", but some assignments swap; we’ll handle both safely.
    int a = 0, b = 0;
    if (sscanf(header[2], "%d %d", &a, &b) != 2) return NULL;

    // Most PGM files use: cols rows
    // We treat first as numCols, second as numRows
    *numCols = a;
    *numRows = b;

    int n = (*numRows) * (*numCols);
    int *pixels = (int*)malloc((size_t)n * sizeof(int));
    if (!pixels) return NULL;

    for (int i = 0; i < n; i++) {
        if (fscanf(in, "%d", &pixels[i]) != 1) {
            free(pixels);
            return NULL;
        }
    }

    return pixels;
}

int pgmWrite(const char **header, const int *pixels, int numRows, int numCols, FILE *out)
{
    if (!header || !pixels || !out || numRows <= 0 || numCols <= 0) return -1;

    // write header lines exactly as stored
    for (int i = 0; i < rowsInHeader; i++) {
        if (fputs(header[i], out) == EOF) return -1;
    }

    // write pixels (ASCII P2). Keep it readable: 20 per line.
    int n = numRows * numCols;
    int count = 0;
    for (int i = 0; i < n; i++) {
        fprintf(out, "%d", pixels[i]);
        count++;
        if (count % 20 == 0 || i == n - 1) {
            fprintf(out, "\n");
        } else {
            fprintf(out, " ");
        }
    }

    return 0;
}

int pgmDrawCircle(int *pixels, int numRows, int numCols,
                  int centerRow, int centerCol, int radius,
                  char **header)
{
    if (!pixels || numRows <= 0 || numCols <= 0 || radius < 0) {
        fprintf(stderr, "pgmDrawCircle: invalid arguments\n");
        return 0;
    }

    int n = numRows * numCols;
    size_t bytes = (size_t)n * sizeof(int);

    int *d_pixels = NULL;
    cudaError_t err = cudaMalloc((void**)&d_pixels, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "pgmDrawCircle cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    err = cudaMemcpy(d_pixels, pixels, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "pgmDrawCircle memcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_pixels);
        return 0;
    }

    launchDrawCircle(d_pixels, numRows, numCols, centerRow, centerCol, radius);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "pgmDrawCircle kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_pixels);
        return 0;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "pgmDrawCircle sync failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_pixels);
        return 0;
    }

    err = cudaMemcpy(pixels, d_pixels, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "pgmDrawCircle memcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_pixels);
        return 0;
    }

    cudaFree(d_pixels);
    return updateHeaderMax(header, pixels, n);
}

int pgmDrawEdge(int *pixels, int numRows, int numCols, int edgeWidth, char **header)
{
    if (!pixels || numRows <= 0 || numCols <= 0 || edgeWidth < 0) {
        fprintf(stderr, "pgmDrawEdge: invalid arguments\n");
        return 0;
    }

    int n = numRows * numCols;
    size_t bytes = (size_t)n * sizeof(int);

    int *d_pixels = NULL;
    cudaError_t err = cudaMalloc((void**)&d_pixels, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "pgmDrawEdge cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    err = cudaMemcpy(d_pixels, pixels, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "pgmDrawEdge memcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_pixels);
        return 0;
    }

    launchDrawEdge(d_pixels, numRows, numCols, edgeWidth);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "pgmDrawEdge kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_pixels);
        return 0;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "pgmDrawEdge sync failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_pixels);
        return 0;
    }

    err = cudaMemcpy(pixels, d_pixels, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "pgmDrawEdge memcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_pixels);
        return 0;
    }

    cudaFree(d_pixels);
    return updateHeaderMax(header, pixels, n);
}

int pgmDrawLine(int *pixels, int numRows, int numCols, char **header,
                int p1row, int p1col, int p2row, int p2col)
{
    if (!pixels || numRows <= 0 || numCols <= 0) {
        fprintf(stderr, "pgmDrawLine: invalid arguments\n");
        return 0;
    }

    int n = numRows * numCols;
    size_t bytes = (size_t)n * sizeof(int);

    int *d_pixels = NULL;
    cudaError_t err = cudaMalloc((void**)&d_pixels, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "pgmDrawLine cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    err = cudaMemcpy(d_pixels, pixels, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "pgmDrawLine memcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_pixels);
        return 0;
    }

    launchDrawLine(d_pixels, numRows, numCols, p1row, p1col, p2row, p2col);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "pgmDrawLine kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_pixels);
        return 0;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "pgmDrawLine sync failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_pixels);
        return 0;
    }

    err = cudaMemcpy(pixels, d_pixels, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "pgmDrawLine memcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_pixels);
        return 0;
    }

    cudaFree(d_pixels);
    return updateHeaderMax(header, pixels, n);
}
