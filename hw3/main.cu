#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

typedef unsigned long long bignum;

static void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

__host__ __device__ int isPrime(bignum x)
{
    if (x == 1) return 0;
    if (x % 2 == 0 && x > 2) return 0;
    bignum i = 2, lim = (bignum) sqrt((float) x) + 1;
    for(; i < lim; i++){
        if (x % i == 0)
            return 0;
    }
    return 1;
}

__global__ void primeKernel(bignum N, int* d_results) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Handle 2 separately
    if (id == 0 && N >= 2) d_results[2] = 1;

    
    bignum num = (bignum)(2ULL * (bignum)id + 1ULL);

    if (num <= N) {
        d_results[num] = isPrime(num);
    }
}

int main(int argc, char* argv[]){

    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <N> <BLOCK_SIZE>\n", argv[0]);
        return 1;
    }

    bignum N = (bignum) std::strtoull(argv[1], nullptr, 10);
    int BLOCK_SIZE = std::atoi(argv[2]);

    if (N <= 0 || BLOCK_SIZE <= 0) {
        std::fprintf(stderr, "N and BLOCK_SIZE must be positive integers.\n");
        std::fprintf(stderr, "N: %llu, BLOCK_SIZE: %d\n",
                     (unsigned long long)N, BLOCK_SIZE);
        return 1;
    }

    // ceil((N + 1) / 2.0 / blockSize)
    int GRID_SIZE = (int) std::ceil(((double)(N + 1ULL)) / 2.0 / (double)BLOCK_SIZE);

    // Host result array: result[i] is 0/1
    size_t arrSize = (size_t)(N + 1ULL) * sizeof(int);
    int* h_results = (int*) std::malloc(arrSize);
    if (!h_results) {
        std::fprintf(stderr, "Failed to allocate host results array.\n");
        return 1;
    }

    // Device result array
    int* d_results = nullptr;
    cudaCheck(cudaMalloc((void**)&d_results, arrSize), "cudaMalloc(d_results)");

    // Initialize device result array to 0
    cudaCheck(cudaMemset(d_results, 0, arrSize), "cudaMemset(d_results)");

    std::printf("Find all prime numbers in the range of 0 to %llu...\n",
                (unsigned long long)N);

    // Launch kernel
    primeKernel<<<GRID_SIZE, BLOCK_SIZE>>>(N, d_results);
    cudaCheck(cudaGetLastError(), "kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // Copy back to host
    cudaCheck(cudaMemcpy(h_results, d_results, arrSize, cudaMemcpyDeviceToHost),
              "cudaMemcpy(d_results -> h_results)");

    
    // Count primes
    unsigned long long primeCount = 0;
    for (bignum i = 0; i <= N; i++) {
        primeCount += (unsigned long long)h_results[i];
    }

    std::printf("Total number of primes found on the GPU in that range is: %llu.\n",
                primeCount);

    // Cleanup
    cudaFree(d_results);
    std::free(h_results);

    return 0;
}
