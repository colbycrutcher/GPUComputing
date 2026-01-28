#include "myKernel.h"

__global__ void kernel( int *a, int dimx, int dimy )
{
    int ix   = blockIdx.x*blockDim.x + threadIdx.x;
    int iy   = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = iy*dimx + ix;

    a[idx]  = a[idx]+1;
}

// Please implement the following kernels2 through kernel6,
// in order to meet the requirements in the write-ups. 
__global__ void kernel2( int *a, int dimx, int dimy )
{
}

__global__ void kernel3( int *a, int dimx, int dimy )
{
}

__global__ void kernel4( int *a, int dimx, int dimy )
{
}

__global__ void kernel5( int *a, int dimx, int dimy )
{
}

__global__ void kernel6( int *a, int dimx, int dimy )
{
}


