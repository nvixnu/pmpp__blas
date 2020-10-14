#include "nvixnu__gemm.h"

__global__
void nvixnu__gemm(float *A, float *B, float * C, const int I, const int J, const int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((col < K) && (row < I)){
        float dot_prod = 0;
        for(int idx = 0; idx < J; idx++){
            dot_prod += A[row*J+idx]*B[idx*K+col];
        }
        C[row*K+col] += dot_prod;
    }
}



__global__
void nvixnu__tiled_gemm(float *A, float *B, float *C, const int I, const int J, const int K, const int TILE_WIDTH){
  // Dinamically allocates the shared memory as a 1D array
  extern __shared__ float shared[];

  // Save the threads and blocks idx into registers
  int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;

  // Calculates the row and col indexes
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  // Element value accumulator
  float dot_prod = 0.0;

  // Strip Mining outter loop. On each phase, a tile of data is fetched and stored in shared memory
  for(int ph = 0; ph < ceil(J/(float)TILE_WIDTH); ph++){
      // Check if the tile is inside the domain 
      if((row < I) && (ph*TILE_WIDTH + tx) < J){
          shared[ty*TILE_WIDTH + tx] = A[row*J + ph*TILE_WIDTH + tx];
      }
      if((col < K) && (ph*TILE_WIDTH + ty) < J){
          shared[TILE_WIDTH*TILE_WIDTH + ty*TILE_WIDTH + tx] = B[(ph*TILE_WIDTH + ty)*K + col];
      }  

      // Wait for all threads in the block to complete the data loading
      __syncthreads();

      // Performs the dot product with the data loaded on this phase
      for(int idx = 0; idx < TILE_WIDTH; idx++){
          dot_prod += shared[ty*TILE_WIDTH + idx]*shared[TILE_WIDTH*TILE_WIDTH + idx*TILE_WIDTH + tx];
      } 

      // Wait for all threads in the block to complete the calculation
      __syncthreads();   
  }

  // Saves the dot product to C[row][col] position
  if((row < I) && (col < K)){
      C[row*K + col] += dot_prod;
  }
}

void nvixnu__h_gemm(float *A, float *B, float *C, const int I,const int J,const int K){
  for(int i = 0; i < I; i++){        
    for(int k = 0; k < K; k++){
      float dot_prod = 0;
        for(int idx = 0; idx < J; idx++){
            dot_prod += A[i*J+idx]*B[idx*K+k];
        }   
        C[i*K+k] += dot_prod;     
      }          
  }    
}