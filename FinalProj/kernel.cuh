//
// Created by ben on 3/16/2026.
//

#ifndef FINALPROJ_GPUMERGESORT_H
#define FINALPROJ_GPUMERGESORT_H


template<uint sortDir, typename T> __device__ uint binarySearchExclusive(uint val, uint *data, uint lo, uint hi)
{
	//ascending order:
	while (lo < hi)
	{
		uint mid = (lo + hi) >> 1;
		if ((sortDir && data[mid] < val) || (!sortDir && data[mid] > val)) // can probably make this branchless?
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo;
}

template<uint sortDir, typename T> __device__ uint binarySearchInclusive(uint val, uint *data, uint lo, uint hi)
{
	//ascending order:
	while (lo < hi)
	{
		uint mid = (lo + hi) >> 1;
		if ((sortDir && data[mid] <= val) || (!sortDir && data[mid] >= val)) // can probably make this branchless?
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo;
}


template<uint sortDir, typename T>
__device__
uint binarySearchExclusive(T key, T *keys, uint lo, uint hi)
{

	while (lo < hi)
	{
		uint mid = (lo + hi) >> 1;
		if ((sortDir && keys[mid] < key) || (!sortDir && keys[mid] > key)) // can probably make this branchless later?
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo;
}

template<uint sortDir, typename T>
 __device__
uint binarySearchInclusive(T key, T *keys, uint lo, uint hi)
{

	while (lo < hi)
	{
		uint mid = (lo + hi) >> 1;
		if ((sortDir && keys[mid] <= key) || (!sortDir && keys[mid] >= key))
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo;
}



//must be called log2n times in order to fully merge
template<uint sortDir, typename T>
__global__
void mergeSortedTilesKernel(T *dkey, uint *dval, T* skey, uint *sval, int N, int tileSize)
{
	//maybe try and cut this up so it uses shared memory
	//we calculate the ranks for 2 elements per thread, meaning we only use half of the required threads.
	//tiles are pretty much seperate from the blocksize, so if we use shared memory, we should be able to convert this pretty easily
	int gx = blockDim.x * blockIdx.x + threadIdx.x;
	int tx = threadIdx.x;
	int tilePairIdx = (gx / (tileSize * 2));
	int pairPos = gx & (tileSize * 2 - 1);

	int indexInTile = gx & (tileSize - 1);

	int startTileA = (tilePairIdx * tileSize * 2);
	int startTileB = (tilePairIdx * tileSize * 2 + tileSize);

	T *a_ptr = skey + startTileA;//moves pointer to start of the first tile
	T *b_ptr = skey + startTileB;//moves pointer to the start of the second tile

	//rank(ai, C) = tileIndex + rank(ai, B) where rank(ai, B) is simply the count of elements bj in B with bj < ai
	// so I guess rank(ai, B) is the index of where ai would be if it were in b. We do this here with both a and b
	// to save threads

	uint lo = 0; uint hi = tileSize;
	uint rankA = indexInTile + binarySearchInclusive<sortDir, T>(*(a_ptr+indexInTile), b_ptr, lo , hi);
	uint rankB = indexInTile + binarySearchExclusive<sortDir, T>(*(b_ptr+indexInTile), a_ptr, lo , hi);

	int indexA = rankA + startTileA;
	int indexB = rankB + startTileA;

	int globalIndexA = indexInTile + startTileA;
	int globalIndexB = indexInTile + startTileB;

	dkey[indexA] = *(a_ptr + indexInTile); // + 2 more global mem accesses
	dkey[indexB] = *(b_ptr + indexInTile);
	dval[indexA] = sval[globalIndexA];
	dval[indexB] = sval[globalIndexB];
	//4 + log(N)*2 global memory accesses
}


#endif //FINALPROJ_GPUMERGESORT_H