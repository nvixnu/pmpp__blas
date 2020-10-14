#ifndef NVIXNU__MM_H_
#define NVIXNU__MM_H_

__global__
void nvixnu__square_mm(float *A, float *B, float * C, int n);

__global__
void vixnu__tiled_square_mm(float *A, float *B, float *C, int n);

#endif /* NVIXNU__MM_H_ */