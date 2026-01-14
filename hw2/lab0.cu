__global__ void kernell(int * a){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    a[idx] = idx;
}

__global__ void kernell(int* b){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    b[idx] = b[idx] * 2;
    
}