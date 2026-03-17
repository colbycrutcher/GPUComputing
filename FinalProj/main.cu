// Main CUDA program
#include <iostream>
//#include "kernel.cu"
#include "kernel.cuh"
#include "util.h"

// Follow steps of last lab with in the main
//Have both kernels write to a output file where it displays the differences between the speed of the two kernels.


template<uint sortDir, typename T>
void populateArrayWithDummyPreSortedTilesForTestingTheMergeKernel(T *key, uint *val, int tileSize, int numTiles, int N)
{
	//creates a mergable array

	if (sortDir == 1)
	    for ( int i = 0, j = 0; i < N; i++)
	    {
	    	if (i % tileSize == 0) j++;
	    	int stride = numTiles;
	    	key[i] = (T)(j + (i % tileSize)*stride);// 0 3 6 9 1 4 7 10 ...
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


int main() {
    const int N = 1 << 10; // Size of the array

	//host keys (data to sort) and values (index to where they are sorted from)
	float *h_key = new float[N];
	uint *h_val = new uint[N];

	//source keys and source values for the GPU
	float *d_dkey, *d_skey;
	uint *d_dval, *d_sval;

    // Initialize host arrays and prepare kernel args
    int blockSize = 256;
 	int numBlocks = (N + blockSize - 1) / blockSize;

	if ( N % numBlocks != 0)
	{
		std::cerr << "bad input. numBlocks must be a factor of N!\n";
		exit(1);
	}

	populateArrayWithDummyPreSortedTilesForTestingTheMergeKernel<1U, float>(h_key, h_val, blockSize, numBlocks, N);

    // Allocate device memory
	cudaMalloc(&d_dkey, sizeof(float) * N);
	cudaMalloc(&d_skey, sizeof(float) * N);
	cudaMalloc(&d_dval, sizeof(uint) * N);
	cudaMalloc(&d_sval, sizeof(uint) * N);

    // Copy data from host to device
	cudaMemcpy(d_skey, h_key, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sval, h_val, sizeof(uint)*N, cudaMemcpyHostToDevice);

	std::cout << "\nunsorted keys\n";
	printArrayTo<float>(std::cout, h_key, N);
	std::cout << "\nunsorted values\n";
	printArrayTo<uint>(std::cout, h_val, N);

	//initial sort

    // Merge after done sorting

	int tileSize = blockSize;

	for (int l = 0 ; tileSize <= N; l++)
	{
		mergeSortedTilesKernel<1U><<<numBlocks, blockSize>>>(d_dkey, d_dval,d_skey, d_sval, N, tileSize);

		//ping ponging between dest and source arrays
		uint *tempVal = d_dval;
		d_dval = d_sval;
		d_sval = tempVal;

		float *tempKey = d_skey;
		d_skey = d_dkey;
		d_dkey = tempKey;

		tileSize *= 2;
	}

    // Copy result back to host
    cudaMemcpy(h_key, d_dkey, sizeof(float)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_val, d_dval, sizeof(uint)*N, cudaMemcpyDeviceToHost);

    // Output results    std::cout << "First 10 results: " << std::endl;
	std::cout << "\nsorted keys\n";
	printArrayTo<float>(std::cout, h_key, N);

	std::cout << "\nsorted values\n";
	printArrayTo<uint>(std::cout, h_val, N);

    // Clean up
    cudaFree(d_dkey);
    cudaFree(d_skey);
    cudaFree(d_dval);
    cudaFree(d_sval);
    delete[] h_key;
    delete[] h_val;
    return 0;
}

