#ifndef NVIXNU__GEMM_H_
#define NVIXNU__GEMM_H_

/**
* Kernel that performs a naive (without tiling) GEMM operation
* @param A The matrix A
* @param B The matrix B
* @param C The matrix C
* @param I The number of rows of the matrix A
* @param J The number of columns of the matrix A (The number of rows of the matrix B)
* @param K The number of columns of the matrix B
*/
__global__ void nvixnu__gemm(double *A, double *B, double * C, const int I, const int J, const int K);

/**
* Kernel that performs a tiled GEMM operation
* @param A The matrix A
* @param B The matrix B
* @param C The matrix C
* @param I The number of rows of the matrix A
* @param J The number of columns of the matrix A (The number of rows of the matrix B)
* @param K The number of columns of the matrix B
* @param TILE_WIDTH The tile width for square tiles
*/
__global__ void nvixnu__tiled_gemm(double *A, double *B, double *C, const int I, const int J, const int K, const int TILE_WIDTH);

/**
* Host function that performs a naive GEMM operation
* @param A The matrix A
* @param B The matrix B
* @param C The matrix C
* @param I The number of rows of the matrix A
* @param J The number of columns of the matrix A (The number of rows of the matrix B)
* @param K The number of columns of the matrix B
*/
void nvixnu__h_gemm(double *A, double *B, double *C, const int I,const int J,const int K);

#endif /* NVIXNU__GEMM_H_ */