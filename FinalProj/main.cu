// Main CUDA program
#include <iostream>
#include "kernel.cu"

// Follow steps of last lab with in the main
//Have both kernels write to a output file where it displays the differences between the speed of the two kernels.
int main() {
    const int N = 1 << 20; // Size of the array
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *d_a, *d_b;

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    kernel<<<numBlocks, blockSize>>>(d_a, d_b, N);

    // Copy result back to host
    cudaMemcpy(h_a, d_a, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Output results    std::cout << "First 10 results: " << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_a[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;
    delete[] h_b;   
    return 0;
}
