#ifndef FINALPROJ_GPUMERGESORT_H
#define FINALPROJ_GPUMERGESORT_H

typedef unsigned int uint;

// ==========================================
// BINARY SEARCH HELPER FUNCTIONS
// ==========================================

template<uint sortDir, typename T> 
__device__ uint binarySearchExclusive(T val, T *data, uint lo, uint hi)
{
    while (lo < hi) {
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
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if ((sortDir && data[mid] <= val) || (!sortDir && data[mid] >= val))
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// ==========================================
// STEP 2: ODD-EVEN TILE SORT KERNEL
// ==========================================

template<uint sortDir, typename T>
__global__ void sortTilesOddEvenKernel(T *dkey, uint *dval, int N, int tileSize) {
    __shared__ T s_keys[1024];
    __shared__ uint s_vals[1024];

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int globalIdx = bx * tileSize + tx;

    if (globalIdx < N && tx < tileSize) {
        s_keys[tx] = dkey[globalIdx];
        s_vals[tx] = dval[globalIdx];
    }
    __syncthreads();

    for (int i = 0; i < tileSize; i++) {
        if (i % 2 == 0) {
            if (tx % 2 == 0 && tx < tileSize - 1 && globalIdx + 1 < N) {
                bool swap = (sortDir == 1) ? (s_keys[tx] > s_keys[tx + 1]) : (s_keys[tx] < s_keys[tx + 1]);
                if (swap) {
                    T tempKey = s_keys[tx]; s_keys[tx] = s_keys[tx + 1]; s_keys[tx + 1] = tempKey;
                    uint tempVal = s_vals[tx]; s_vals[tx] = s_vals[tx + 1]; s_vals[tx + 1] = tempVal;
                }
            }
        } else {
            if (tx % 2 != 0 && tx < tileSize - 1 && globalIdx + 1 < N) {
                bool swap = (sortDir == 1) ? (s_keys[tx] > s_keys[tx + 1]) : (s_keys[tx] < s_keys[tx + 1]);
                if (swap) {
                    T tempKey = s_keys[tx]; s_keys[tx] = s_keys[tx + 1]; s_keys[tx + 1] = tempKey;
                    uint tempVal = s_vals[tx]; s_vals[tx] = s_vals[tx + 1]; s_vals[tx + 1] = tempVal;
                }
            }
        }
        __syncthreads();
    }

    if (globalIdx < N && tx < tileSize) {
        dkey[globalIdx] = s_keys[tx];
        dval[globalIdx] = s_vals[tx];
    }
}

// ==========================================
// STEP 3: BINARY SEARCH MERGE KERNEL
// ==========================================

template<uint sortDir, typename T>
__global__ void mergeSortedTilesKernel(T *dkey, uint *dval, T* skey, uint *sval, int N, int tileSize)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gx >= N) return; 

    int tileIdx = gx / tileSize;
    int isRightTile = tileIdx & 1; 
    int pairIdx = tileIdx / 2;

    int startA = pairIdx * tileSize * 2;
    int startB = startA + tileSize;
    
    int sizeA = (N - startA > tileSize) ? tileSize : N - startA;
    if (sizeA < 0) sizeA = 0;
    int sizeB = (N - startB > tileSize) ? tileSize : N - startB;
    if (sizeB < 0) sizeB = 0;

    T element = skey[gx];
    uint value = sval[gx];
    int outIdx;

    if (!isRightTile) {
        uint rankB = binarySearchExclusive<sortDir, T>(element, skey + startB, 0, sizeB);
        uint rankA = gx - startA;
        outIdx = startA + rankA + rankB;
    } else {
        uint rankA = binarySearchInclusive<sortDir, T>(element, skey + startA, 0, sizeA);
        uint rankB = gx - startB;
        outIdx = startA + rankA + rankB;
    }

    dkey[outIdx] = element;
    dval[outIdx] = value;
}

#endif //FINALPROJ_GPUMERGESORT_H