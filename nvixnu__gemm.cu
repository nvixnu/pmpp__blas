#include "nvixnu__gemm.h"

__global__
void nvixnu__square_mm(float *A, float *B, float * C, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((col < n) && (row < n)){
        float dot_prod = 0;
        for(int k = 0; k < n; k++){
            dot_prod += A[row*n+k]*B[k*n+col];
        }
        C[row*n+col] += dot_prod;
    }
}

__global__
void vixnu__tiled_square_mm(float *A, float *B, float *C, int n){
  // Dinamically allocates the shared memory as a 1D array
  extern __shared__ float shared[];

  // Save the threads and blocks idx into registers
  int tile_size = blockDim.x;
  int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;

  // Calculates the row and col indexes
  int row = by * tile_size + ty;
  int col = bx * tile_size + tx;

  // Element value accumulator
  float dot_prod = 0.0;

  // Strip Mining outter loop. On each phase, a tile of data is fetched and stored in shared memory
  for(int ph = 0; ph < ceil(n/(float)tile_size); ph++){
      // Check if the tile is inside the domain 
      if((row < n) && (ph*tile_size + tx) < n){
          shared[ty*tile_size + tx] = A[row*n + ph*tile_size + tx];
      }
      if((col < n) && (ph*tile_size + ty) < n){
          shared[tile_size*tile_size + ty*tile_size + tx] = B[(ph*tile_size + ty)*n + col];
      }  

      // Wait for all threads in the block to complete the data loading
      __syncthreads();

      // Performs the dot product with the data loaded on this phase
      for(int k = 0; k < tile_size; k++){
          dot_prod += shared[ty*tile_size + k]*shared[tile_size*tile_size + k*tile_size + tx];
      } 

      // Wait for all threads in the block to complete the calculation
      __syncthreads();   
  }

  // Saves the dot product to C[row][col] position
  if((row < n) && (col < n)){
      C[row*n + col] += dot_prod;
  }
}