#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include "timing.h"


typedef unsigned long long bignum;

static void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
         fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
         exit(1);
    }
}

__host__ __device__ int isPrime(bignum x)
{
    if (x == 1) return 0;
    if (x % 2 == 0 && x > 2) return 0;
    bignum i = 2, lim = (bignum) sqrt((float) x) + 1;
    for(; i < lim; i++){
        if (x % i == 0) return 0;
    }
    return 1;
}

__global__ void primeKernel(bignum N, int* d_results) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Handle 2 separately
    if (id == 0 && N >= 2) d_results[2] = 1;

    // Thread id -> odd number: 1,3,5,7,...
    bignum num = (bignum)(2ULL * (bignum)id + 1ULL);

    if (num <= N) {
        d_results[num] = isPrime(num);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
         fprintf(stderr, "Usage: %s <N> <BLOCK_SIZE>\n", argv[0]);
        return 1;
    }

    bignum N = (bignum)  strtoull(argv[1], nullptr, 10);
    int BLOCK_SIZE =  atoi(argv[2]);

    if (N == 0 || BLOCK_SIZE <= 0) {
         fprintf(stderr, "N and BLOCK_SIZE must be positive integers.\n");
         fprintf(stderr, "N: %llu, BLOCK_SIZE: %d\n",
                     (unsigned long long)N, BLOCK_SIZE);
        return 1;
    }

    int GRID_SIZE = (int)  ceil(((double)(N + 1ULL)) / 2.0 / (double)BLOCK_SIZE);

    size_t arrSize = (size_t)(N + 1ULL) * sizeof(int);

    int* h_results = (int*)  malloc(arrSize);
    int* h_serial  = (int*)  malloc(arrSize);
    if (!h_results || !h_serial) {
         fprintf(stderr, "Failed to allocate host arrays.\n");
        return 1;
    }

     printf("Find all prime numbers in the range of 0 to %llu...\n",
                (unsigned long long)N);

    double pStart = currentTime();

    int* d_results = nullptr;
    cudaCheck(cudaMalloc((void**)&d_results, arrSize), "cudaMalloc(d_results)");
    cudaCheck(cudaMemset(d_results, 0, arrSize), "cudaMemset(d_results)");

    primeKernel<<<GRID_SIZE, BLOCK_SIZE>>>(N, d_results);
    cudaCheck(cudaGetLastError(), "kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    cudaCheck(cudaMemcpy(h_results, d_results, arrSize, cudaMemcpyDeviceToHost),
              "cudaMemcpy(d_results -> h_results)");

    double pEnd = currentTime();
    double parallelTime = pEnd - pStart;

    // Count primes on CPU AFTER timing (exclude summation cost)
    unsigned long long gpuPrimeCount = 0;
    for (bignum i = 0; i <= N; i++) gpuPrimeCount += (unsigned long long)h_results[i];

     printf("Parallel code executiontime in seconds is %.6f\n", parallelTime);
     printf("Total number of primes found on the GPU in that range is: %llu.\n",
                gpuPrimeCount);

    // Cleanup device now that we have results
    cudaFree(d_results);

memset(h_serial, 0, arrSize);

double sStart = currentTime();

/* Mark primes on CPU */
if (N >= 2) h_serial[2] = 1;
for (bignum x = 1; x <= N; x += 2) {
    h_serial[x] = isPrime(x);
}

double sEnd = currentTime();
double serialTime = sEnd - sStart;

/* Count AFTER timing */
unsigned long long cpuPrimeCount = 0;
for (bignum i = 0; i <= N; i++) {
    cpuPrimeCount += (unsigned long long)h_serial[i];
}

printf("Serial code executiontime in seconds is %.6f\n", serialTime);
printf("Total number of primes by CPU in that range is: %llu.\n",
       cpuPrimeCount);



    // Mark primes in [0..N]
    if (N >= 2) h_serial[2] = 1;
    for (bignum x = 1; x <= N; x += 2) {   // only odds (1,3,5,...)
        h_serial[x] = isPrime(x);
    }

    // Sum AFTER timing
    for (bignum i = 0; i <= N; i++) cpuPrimeCount += (unsigned long long)h_serial[i];

    printf("Serial code execution time in seconds is %.6f\n", serialTime);
     printf("Total number of primes by CPU in that range is: %llu.\n",
                cpuPrimeCount);

    // Speedup/efficiency lines (match sample format)
    const double speedup = serialTime / parallelTime;
    const double NumProcessorCores = 4.0;
    const double efficiency = speedup / NumProcessorCores;

     printf("%%%%%% The speedup(SerialTimeCost / ParallelTimeCost) when using GPU is %.6f\n", speedup);
     printf("%%%%%% The efficiency(Speedup / NumProcessorCores) when using GPU is %.6f\n", efficiency);

     free(h_results);
     free(h_serial);
    return 0;
}
