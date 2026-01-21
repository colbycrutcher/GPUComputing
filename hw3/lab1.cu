#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <cmath>
#include <stdlib.h>


typedef long long int bignum;
/*
__global__ void addKernel(int *a, int *b, int *c, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
*/
__host__ __device__ int isPrime(bignum n){
    if (n <= 1) return 0;
    if (n % 2 == 0 && n > 2) return 0;

    bignum i = 2, lim(bignum) sqrt((float)n) + 1;

    for (;i <= lim; i++){
        if (n % i == 0) return 0;
    }   
    
    return 1;
}

int main(int argc, char* argv[]){
    //Command line arguments for N and block size
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <N> <BLOCK_SIZE>" << std::endl;
        return 1;
    
    }

    int N = atoi(argv[1]);
    int BLOCK_SIZE = atoi(argv[2]);

    //Check for valid N and BLOCK_SIZE
    if (N <= 0 || BLOCK_SIZE <= 0) {
        std::cerr << "N and BLOCK_SIZE must be positive integers." << std::endl;
        return 1;
    }

    int GRID_SIZE = (int) ceil((N + 1)/ 2.0 / BLOCK_SIZE);

    return 0;
}
