#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "myKernel.h"

void printArr(int *a, char *name, int dimy, int dimx);


int main()
{
    int dimx = 16;
    int dimy = 16;
    int num_bytes = dimx*dimy*sizeof(int);

    int *d_a=0, *h_a=0; // device and host pointers

    h_a = (int*)malloc(num_bytes);
    cudaMalloc( (void**)&d_a, num_bytes );

    if( 0==h_a || 0==d_a )
    {
        printf("couldn't allocate memory\n");
        return 1;
    }

    cudaMemset( d_a, 0, num_bytes );

    dim3 grid, block;
    block.x = 3;
    block.y = 4;
    grid.x  = ceil( (float)dimx / block.x );
    grid.y  = ceil( (float)dimy / block.y );
    
    // Use kernel to fill d_a array
    kernel<<<grid, block>>>( d_a, dimx, dimy );
    cudaMemcpy( h_a, d_a, num_bytes, cudaMemcpyDeviceToHost );
    printArr(h_a, "kernel ", dimy, dimx);

    // Use kernel2 to fill d_a array
    cudaMemset( d_a, 0, num_bytes );
    kernel2<<<grid, block>>>( d_a, dimx, dimy );
    cudaMemcpy( h_a, d_a, num_bytes, cudaMemcpyDeviceToHost );
    printArr(h_a, "kernel 2", dimy, dimx);

    // Use kernel3 to fill d_a array
    cudaMemset( d_a, 0, num_bytes );
    kernel3<<<grid, block>>>( d_a, dimx, dimy );
    cudaMemcpy( h_a, d_a, num_bytes, cudaMemcpyDeviceToHost );
    printArr(h_a, "kernel 3", dimy, dimx);

    // Use kernel4 to fill d_a array
    cudaMemset( d_a, 0, num_bytes );
    kernel4<<<grid, block>>>( d_a, dimx, dimy );
    cudaMemcpy( h_a, d_a, num_bytes, cudaMemcpyDeviceToHost );
    printArr(h_a, "kernel 4", dimy, dimx);
   
    // Use kernel5 to fill d_a array
    cudaMemset( d_a, 0, num_bytes );
    kernel5<<<grid, block>>>( d_a, dimx, dimy );
    cudaMemcpy( h_a, d_a, num_bytes, cudaMemcpyDeviceToHost );
    printArr(h_a, "kernel 5", dimy, dimx);
   
    // Use kernel6 to fill d_a array
    cudaMemset( d_a, 0, num_bytes );
    kernel6<<<grid, block>>>( d_a, dimx, dimy );
    cudaMemcpy( h_a, d_a, num_bytes, cudaMemcpyDeviceToHost );
    printArr(h_a, "kernel 6", dimy, dimx);

    free( h_a );
    cudaFree( d_a );

    return 0;
}

// print array 
void printArr(int *a, char *name, int dimy, int dimx)
{
    if(name == NULL)
        return;
    printf("===================================%s====================================\n", name);
    for(int row=0; row<dimy; row++)
    {
        for(int col=0; col<dimx; col++)
            printf("%-4d ", a[row*dimx+col] );
        printf("\n");
    }
    printf("==============================================================================\n");
}


