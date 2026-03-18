#ifndef FINALPROJ_GPUMERGESORT_H
#define FINALPROJ_GPUMERGESORT_H

typedef unsigned int uint;

template<uint sortDir, typename T> 
__device__ uint binarySearchExclusive(T val, T *data, uint lo, uint hi)
{
    while (lo < hi)
    {
        uint mid = (lo + hi) >> 1;
        if ((sortDir && data[mid] < val) || (!sortDir && data[mid] > val))
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

template<uint sortDir, typename T> 
__device__ uint binarySearchInclusive(T val, T *data, uint lo, uint hi)
{
    while (lo < hi)
    {
        uint mid = (lo + hi) >> 1;
        if ((sortDir && data[mid] <= val) || (!sortDir && data[mid] >= val))
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// Global memory merge kernel
template<uint sortDir, typename T>
__global__
void mergeSortedTilesKernel(T *dkey, uint *dval, T* skey, uint *sval, int N, int tileSize)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gx >= N) return; // Bounds check

    int tileIdx = gx / tileSize;
    int isRightTile = tileIdx & 1; // 0 if A (left), 1 if B (right)
    int pairIdx = tileIdx / 2;

    int startA = pairIdx * tileSize * 2;
    int startB = startA + tileSize;
    
    // Safety bounds for incomplete tiles
    int sizeA = (N - startA > tileSize) ? tileSize : N - startA;
    if (sizeA < 0) sizeA = 0;
    int sizeB = (N - startB > tileSize) ? tileSize : N - startB;
    if (sizeB < 0) sizeB = 0;

    T element = skey[gx];
    uint value = sval[gx];
    int outIdx;

    if (!isRightTile) {
        // Element is in A
        uint rankB = binarySearchExclusive<sortDir, T>(element, skey + startB, 0, sizeB);
        uint rankA = gx - startA;
        outIdx = startA + rankA + rankB;
    } else {
        // Element is in B 
        uint rankA = binarySearchInclusive<sortDir, T>(element, skey + startA, 0, sizeA);
        uint rankB = gx - startB;
        outIdx = startA + rankA + rankB;
    }

    dkey[outIdx] = element;
    dval[outIdx] = value;
}

#endif 