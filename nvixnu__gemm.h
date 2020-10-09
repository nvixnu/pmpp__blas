#ifndef NVIXNU__GEMM_H_
#define NVIXNU__GEMM_H_

__global__
void nvixnu__gemm_square(float *A, float *B, float * C, int n);

__global__
void vixnu__tiled_gemm_square(float *A, float *B, float *C, int n, int tile_size);

#endif /* NVIXNU__GEMM_H_ */