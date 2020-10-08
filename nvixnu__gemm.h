#ifndef NVIXNU__GEMM_H_
#define NVIXNU__GEMM_H_

__global__
void nvixnu__gemm(float *A, float *B, float * C, int n);

#endif /* NVIXNU__GEMM_H_ */