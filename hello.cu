#include <stdio.h>

__global__ void hello(){

  printf("Hello from block: %u, thread: %u\n", threadIdx.x, blockIdx.x);
}

int main(){

  hello<<<2,1>>>();
  cudaDeviceSynchronize();
}

