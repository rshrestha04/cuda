#include<stdio.h>
#include "device_launch_parameters.h"

const int DSIZE =8192;
const int block_size =32;
const float A_val = 3.0f;
const float B_val = 2.0f;


//matrix addition in CUDA

__global__ void madd(const float *A, const float *B, const float *C, int ds  ){
  int idx = threadIDx.x + blockDim.x*blockIDx.x; 
  int idy = threadIDx.y + blockDim.y*blockIDx.y;
  int idnx = ds* idy + idx; 	
   
    if (idx < ds && idy < ds){
   	C[idnx] = A[idnx]+ B[idnx];
    }  
}


int main(){
float *h_A , *h_B, *h_C, *d_A, *d_B, *d_C ;


h_A = new float[DSIZE*DSIZE];
h_B = new float[DSIZE*DSIZE];
h_C = new float[DSIZE*DSIZE];

  for(int i = 0; i < DSIZE*DSIZE; i++){
  h_A[i] = A_val;
  h_B[i] = B_val; 
  h_C[i] = 0;
  }

cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(float));
cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(float));
//cudaCheckErrors("CudaMalloc faliure");
cudaMemcpy(d_A, h_A,DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost );
cudaMemcpy(d_B, h_B,DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost );
//cudaCheckErrors("CudaMemcpy H2D faliure");

  dim3 block(block_size, block_size);  // dim3 variable holds 3 dimensions
  dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);
  madd<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
  //cudaCheckErrors("kernel launch failure");
  // Cuda processing sequence step 2 is complete
  // Copy results back to host
  cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

}
}
