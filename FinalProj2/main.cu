// Main CUDA program
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <string>
#include "kernel.cuh"
#include "util.h"

// Forward declaration for CPU sort
void sequentialMergeSort(float* data, int size);

template<uint sortDir, typename T>
void populateArrayWithDummyPreSortedTilesForTestingTheMergeKernel(T *key, uint *val, int tileSize, int numTiles, int N)
{
    if (sortDir == 1)
        for ( int i = 0, j = 0; i < N; i++)
        {
            if (i % tileSize == 0) j++;
            int stride = numTiles;
            key[i] = j + (i % tileSize)*stride;
            val[i] = static_cast<uint>(i);
        }
    else
        for (int i = 0, j = 0; i < N; i++)
        {
            if (i % tileSize == 0) j++;
            int stride = numTiles;
            int localIdx = i % tileSize;
            int reversedIdx = tileSize - 1 - localIdx;

            key[i] = j + reversedIdx * stride;
            val[i] = static_cast<uint>(i);
        }
}

int main(int argc, char* argv[]) {
    // Default values
    int N = 1 << 10;       // Default size: 1024
    int blockSize = 256;   // Default block size: 256

    // Parse command line arguments
    if (argc > 1) {
        if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
            std::cout << "Usage: ./mergesort [ArraySize] [BlockSize]\n";
            std::cout << "Example: ./mergesort 1048576 256\n";
            return 0;
        }
        N = std::atoi(argv[1]);
    }
    if (argc > 2) {
        blockSize = std::atoi(argv[2]);
    }

    // Validation
    if (N < 2) {
        std::cerr << "Error: Array size (N) must be at least 2.\n";
        return 1;
    }
    if (blockSize < 2 || blockSize > 1024) {
        std::cerr << "Error: Block size must be between 2 and 1024.\n";
        return 1;
    }

    int numBlocks = (N + blockSize - 1) / blockSize;

    if (N % numBlocks != 0) {
        std::cerr << "Error: Bad input. numBlocks must be a factor of N!\n";
        return 1;
    }

    std::cout << "Running Merge Sort with N = " << N << ", Block Size = " << blockSize << "\n";
    std::cout << "------------------------------------------------------\n";

    // Allocate host memory
    float *h_key = new float[N];
    uint *h_val = new uint[N];

    // Populate the array with dummy pre-sorted tiles
    populateArrayWithDummyPreSortedTilesForTestingTheMergeKernel<1U, float>(h_key, h_val, blockSize, numBlocks, N);

    // --- CPU SORT TIMING ---
    float* h_key_cpu = new float[N];
    std::copy(h_key, h_key + N, h_key_cpu);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    sequentialMergeSort(h_key_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = end_cpu - start_cpu;

    // --- GPU MEMORY SETUP ---
    float *d_dkey, *d_skey;
    uint *d_dval, *d_sval;

    cudaMalloc(&d_dkey, sizeof(float) * N);
    cudaMalloc(&d_skey, sizeof(float) * N);
    cudaMalloc(&d_dval, sizeof(uint) * N);
    cudaMalloc(&d_sval, sizeof(uint) * N);

    cudaMemcpy(d_skey, h_key, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sval, h_val, sizeof(uint)*N, cudaMemcpyHostToDevice);

    std::cout << "\nUnsorted keys (first 16):\n";
    printArrayTo<float>(std::cout, h_key, std::min(N, 16));

    // --- GPU SORT TIMING ---
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);

    int tileSize = blockSize;

    for (int l = 0 ; tileSize <= N; l++)
    {
        mergeSortedTilesKernel<1U><<<numBlocks, blockSize>>>(d_dkey, d_dval, d_skey, d_sval, N, tileSize);

        // Ping ponging between dest and source arrays
        uint *tempVal = d_dval; d_dval = d_sval; d_sval = tempVal;
        float *tempKey = d_skey; d_skey = d_dkey; d_dkey = tempKey;

        tileSize *= 2;
    }

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start_gpu, stop_gpu);

    // Copy result back to host (Because of the ping-pong swap at the end of the loop, the final sorted data is in d_skey)
    cudaMemcpy(h_key, d_skey, sizeof(float)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_val, d_sval, sizeof(uint)*N, cudaMemcpyDeviceToHost);

    std::cout << "\nSorted keys (first 16):\n";
    printArrayTo<float>(std::cout, h_key, std::min(N, 16)); 

    // --- WRITE TO FILE ---
    // Using append mode so multiple test runs don't overwrite each other
    std::ofstream outfile("sort_comparison.txt", std::ios_base::app);
    if (outfile.is_open()) {
        outfile << "Array Size (N): " << N << " | Block Size: " << blockSize << "\n";
        outfile << "CPU Sequential Merge Sort Time: " << cpu_ms.count() << " ms\n";
        outfile << "GPU Parallel Merge Sort Time: " << gpu_ms << " ms\n";
        outfile << "Speedup: " << cpu_ms.count() / gpu_ms << "x\n";
        outfile << "--------------------------------------------------\n";
        outfile.close();
        std::cout << "\nSuccessfully appended timing results to sort_comparison.txt\n";
    }

    // Clean up
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    cudaFree(d_dkey);
    cudaFree(d_skey);
    cudaFree(d_dval);
    cudaFree(d_sval);
    delete[] h_key;
    delete[] h_val;
    delete[] h_key_cpu;
    return 0;
}