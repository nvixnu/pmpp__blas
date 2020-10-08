#include "nvixnu__gemm.h"

__global__
void nvixnu__gemm(float *A, float *B, float * C, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((col < n) && (row < n)){
        float dot_prod = 0.0;
        for(int k = 0; k < n; k++){
            dot_prod += A[row*n+k]*B[k*n+col] + C[row*n+col];
        }
        C[row*n+col] = dot_prod;
    }
}