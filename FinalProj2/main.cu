// Main CUDA program
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <string>
#include <algorithm>
#include "kernel.cuh"
#include "util.h"

//declaration for CPU sort
void sequentialMergeSort(float* data, int size);

int main(int argc, char* argv[]) {
    // Default values
    int N = 1 << 10;       // Default size: 1024
    int blockSize = 256;   // Default block size: 256

    // parse command line arguments
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

    std::cout << "Running FULL Merge Sort with N = " << N << ", Block Size = " << blockSize << "\n";
    std::cout << "------------------------------------------------------\n";

    // Allocate host memory
    float *h_key = new float[N];
    uint *h_val = new uint[N];

    //Generate random data 
    srand(42); // Seed for reproducibility during testing
    for (int i = 0; i < N; i++) {
        h_key[i] = static_cast<float>(rand() % 1000); 
        h_val[i] = static_cast<uint>(i);
    }

    // write the unsorted file
    std::ofstream unsortedFile("unsorted.txt");
    if (unsortedFile.is_open()) {
        for (int i = 0; i < N; i++) {
            unsortedFile << h_key[i] << "\n";
        }
        unsortedFile.close();
        std::cout << "Saved unsorted array to unsorted.txt\n";
    }

    // --- CPU SORT TIMING ---
    float* h_key_cpu = new float[N];
    std::copy(h_key, h_key + N, h_key_cpu);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    sequentialMergeSort(h_key_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = end_cpu - start_cpu;

    // --- WRITE CPU RESULTS TO FILE ---
    std::ofstream cpuFile("cpu_sorted.txt");
    if (cpuFile.is_open()) {
        for (int i = 0; i < N; i++) {
            cpuFile << h_key_cpu[i] << "\n";
        }
        cpuFile.close();
        std::cout << "Saved CPU sorted array to cpu_sorted.txt\n";
    }

    // gpu memory
    float *d_dkey, *d_skey;
    uint *d_dval, *d_sval;

    cudaMalloc(&d_dkey, sizeof(float) * N);
    cudaMalloc(&d_skey, sizeof(float) * N);
    cudaMalloc(&d_dval, sizeof(uint) * N);
    cudaMalloc(&d_sval, sizeof(uint) * N);

    // copy initial random data into d_skey to start
    cudaMemcpy(d_skey, h_key, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sval, h_val, sizeof(uint)*N, cudaMemcpyHostToDevice);

    // gpu timer
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);

    //  the initial tiles in parallel
    sortTilesOddEvenKernel<1U><<<numBlocks, blockSize>>>(d_skey, d_sval, N, blockSize);

    // merge the tiles
    int tileSize = blockSize;

    while (tileSize < N)
    {
        mergeSortedTilesKernel<1U><<<numBlocks, blockSize>>>(d_dkey, d_dval, d_skey, d_sval, N, tileSize);

        // swap pointers so the output of this run becomes the input for the next
        uint *tempVal = d_dval; d_dval = d_sval; d_sval = tempVal;
        float *tempKey = d_skey; d_skey = d_dkey; d_dkey = tempKey;

        tileSize *= 2;
    }

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start_gpu, stop_gpu);

    // copy final sorted data back to host
    cudaMemcpy(h_key, d_skey, sizeof(float)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_val, d_sval, sizeof(uint)*N, cudaMemcpyDeviceToHost);

    // write to gpu sorted file
    std::ofstream gpuFile("gpu_sorted.txt");
    if (gpuFile.is_open()) {
        for (int i = 0; i < N; i++) {
            gpuFile << h_key[i] << "\n";
        }
        gpuFile.close();
        std::cout << "Saved GPU sorted array to gpu_sorted.txt\n";
    }

    // write timing
    std::ofstream outfile("sort_comparison.txt", std::ios_base::app);
    if (outfile.is_open()) {
        outfile << "Array Size (N): " << N << " | Block Size: " << blockSize << "\n";
        outfile << "CPU Sequential Merge Sort Time: " << cpu_ms.count() << " ms\n";
        outfile << "GPU Parallel Merge Sort Time: " << gpu_ms << " ms\n";
        outfile << "Speedup: " << cpu_ms.count() / gpu_ms << "x\n";
        outfile << "--------------------------------------------------\n";
        outfile.close();
        std::cout << "Appended timing results to sort_comparison.txt\n";
    }

    // clean up
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