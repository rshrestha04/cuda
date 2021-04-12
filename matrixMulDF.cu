#include <stdio.h>

// these are just for timing measurments
#include <time.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

const int block_size = 4;  // CUDA maximum is 1024 *total* threads in block
//const float A_val = 3.0f;
//const float B_val = 2.0f;

// matrix multiply (naive) kernel: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, int r1,int c1, int r2, int c2) {

  int idx = threadIdx.x+blockDim.x*blockIdx.x; // create thread x index
  int idy = threadIdx.y+blockDim.y*blockIdx.y; // create thread y index

if ((idx < c2) && (idy < c1)){
    float temp = 0;
    for (int i = 0; i < c1; i++)
      temp += A[idy*c1+i] * B[i*c2+idx];   // dot product of row and column
    C[idy*c2+idx] = temp;
  }
}

int main(){

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;


  // these are just for timing
  clock_t t0, t1, t2;
  double t1sum=0.0;
  double t2sum=0.0;

  // start timing
  t0 = clock();

//getting matrix from user

int r1,c1,r2,c2;
printf("Enter the row of 1st Matrix: ");
scanf("%d",&r1);
printf("Enter the column of 1st Matrix: ");
scanf("%d",&c1);
printf("Enter the row of 2st Matrix: ");
scanf("%d",&r2);
printf("Enter the column of 2st Matrix: ");
scanf("%d",&c2); 

if (c1!=r2){
printf("Invalid Matrix");
}

  h_A = new float[r1*c1];
  h_B = new float[r2*c2];
  h_C = new float[r1*c2];
 
FILE* matrixA;

matrixA = fopen("A.txt","r");
if(matrixA==NULL){
printf("Matrix A did not open \n"); 
return 0 ;
}

int i =0;
while(fscanf(matrixA,"%f", &h_A[i] )!= EOF){
i++;
}


FILE* matrixB;
matrixB = fopen("B.txt","r");
if(matrixA==NULL){
printf("Matrix B did not open \n");
return 0;
}

i =0;
while(fscanf(matrixB,"%f",&h_B[i] )!= EOF){
i++;
}


//Printing values on screen for degugg
/*
for (int i = 0; i < r1*c1; i++){
        printf("%.1f ", h_A[i]);
}
printf("\n");
for (int i = 0; i < r2*c2; i++){
        printf("%.1f ", h_B[i]);
}
*/

//If values was assigned in the program and not through input files
/*
 for (int i = 0; i < r1*c1; i++){
    h_A[i] = A_val;
}

 for (int i = 0; i < r2*c2; i++){
    h_B[i] = B_val;
}

 for (int i = 0; i < r1*c2; i++){
    h_C[i] = 0;}
*/

  // Initialization timing
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Init took %f seconds.  Begin compute\n", t1sum);

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, r1*c1*sizeof(float));
  cudaMalloc(&d_B, r2*c2*sizeof(float));
  cudaMalloc(&d_C, r1*c2*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, r1*c1*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, r2*c2*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // Cuda processing sequence step 1 is complete

  // Launch kernel
  dim3 block(block_size, block_size);  // dim3 variable holds 3 dimensions
  dim3 grid(((c1+r2)/2+block.x-1)/block.x, ((c1+r2)/2+block.y-1)/block.y);
  mmul<<<grid, block>>>(d_A, d_B, d_C, r1,c1,r2,c2 );
  cudaCheckErrors("kernel launch failure");

  // Cuda processing sequence step 2 is complete

  // Copy results back to host
  cudaMemcpy(h_C, d_C, r1*c2*sizeof(float), cudaMemcpyDeviceToHost);

for (int i = 0; i < r1*c1; i++){
        printf("%.1f ", h_A[i]);
}
printf("\n");
for (int i = 0; i < r2*c2; i++){
        printf("%.1f ", h_B[i]);
}
printf("\n");
// Verify results
  for (int i = 0; i < r1*c2; i++){
        printf("%.1f ", h_C[i]);
        }


FILE* matrixC;
matrixC = fopen("C.txt","w");
if(matrixC==NULL){
printf("Matrix A did not open \n");
return 0 ;
}

for(int i =0; i < r1*c2; i++){ 
fprintf(matrixC, "%.1f ",h_C[i]);
}

fclose(matrixA);
fclose(matrixB);
fclose(matrixC);


  // GPU timing
  t2 = clock();
  t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
  printf ("Done. Compute took %f seconds\n", t2sum);

  // Cuda processing sequence step 3 is complete

  // Verify results
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  return 0;
}
  
