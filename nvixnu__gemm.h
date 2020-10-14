#ifndef NVIXNU__GEMM_H_
#define NVIXNU__GEMM_H_

__global__
void nvixnu__gemm(float *A, float *B, float * C, const int I, const int J, const int K);

__global__
void nvixnu__tiled_gemm(float *A, float *B, float *C, const int I, const int J, const int K, const int TILE_WIDTH);

void nvixnu__h_gemm(float *A, float *B, float *C, const int I,const int J,const int K);

#endif /* NVIXNU__GEMM_H_ */