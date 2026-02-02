#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>


//kernel1
__global__ void kernel1(int *a)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = 7;
}


//Kernel 2
__global__ void kernel2(int *a)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = blockIdx.x;
}




//Kernel 3
__global__ void kernel3(int *a)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = threadIdx.x;
}


int main(int argc, char* argv[]){

    //Size of vectors
    int n = 16;

    //Host input vectors
    int *h_a;

    //Device input vectors
    int *d_a;

    //Size, in bytes, of each vector
    size_t bytes = n*sizeof(int);

    //Allocate memory for each vector on host
    h_a = (int*)malloc(bytes);

    //Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);

    int i;
    //Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = 0;
    }

    //Copy host vectors to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);

    //Define blocksize and gridsize
    // Number of threads in each block
    int blocksize = 4;
    // Number of blocks in grid
    int gridsize = (int)ceil((float)n/blocksize);


    //Launch the kernels
    kernel1<<<gridsize, blocksize>>>(d_a);

    // Print results after kernel1
    cudaMemcpy( h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    printf("After kernel1:\n");
    for(i=0; i< n; i++){
        printf("%d ",h_a[i]);
    }
    printf("\n");
    // Launch kernel2
    kernel2<<<gridsize, blocksize>>>(d_a);

    // Print results after kernel2
    cudaMemcpy( h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    printf("After kernel2:\n");
    for(i=0; i< n; i++){
        printf("%d ",h_a[i]);
    }
    printf("\n");

    kernel3<<<gridsize, blocksize>>>(d_a);

    // Print results after kernel3
    cudaMemcpy( h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    printf("After kernel3:\n");
    for(i=0; i< n; i++){
        printf("%d ",h_a[i]);
    }
    printf("\n");

    //free GPU memory
    cudaFree(d_a);  
    //free CPU memory
    free(h_a);
    
    return 0;

}