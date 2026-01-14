include <stdio.h>

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


int main(){
    kernel1<<<2,5>>>();
    kernel2<<<2,5>>>();
    kernel3<<<2,5>>>();
}