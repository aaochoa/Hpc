//============================================================================
#include<cstdlib>
#include<time.h>
#include<cuda.h>
#include<iostream>

#define BLOCK_SIZE 1024 // Because it's just an array, 1 dimension

using namespace std;
//====== Serial vector ADD =====================================================
float serialVectorItemsAdd (float *A, int length)
{
  float sum=0;

  for (int i = 0; i < length; i++)
  {
    sum = sum + A[i];
  }
  return sum;
}

//====== Function made to print vector =========================================
void printVector (float *A, int length)
{
  for (int i=0; i<length; i++)
  {
    cout<<A[i]<<" | ";
  }
  cout<<endl;
}

//====== Function made to fill the vector with some given value ================
void fillVector (float *A, float value, int length)
{
  for (int i=0; i<length; i++)
  {
    A[i] = value;
  }
}

//====== To compare both results parallel and serial ===========================
void resultCompare(float A, float  *B)
{
  if(fabs(A-B[0]) < 0.1)
  {
    cout<<"Well Done"<<endl;
  } else
  {
    cout<<"Not working"<<endl;
  }
}

//======= Reduction kernel =====================================================
//Parallel
__global__ void reduceKernel(float *g_idata, float *g_odata, int length)
{
  __shared__ float sdata[BLOCK_SIZE];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  if(i<length){
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
  } else
  {
    sdata[tid] = 0.0;
  }
  __syncthreads();
  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
  {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0)
  {
    g_odata[blockIdx.x] = sdata[0];

  }
}
//====== Function made to call the reduction kernel ============================
void vectorItemsAdd(float *A, float *B, int length)
{
  float * d_A;
  float * d_B;
  float * algo = (float *) malloc(length * sizeof(float));

  cudaMalloc(&d_A,length*sizeof(float));
  cudaMalloc(&d_B,length*sizeof(float));

  cudaMemcpy(d_A, A,length*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B,length*sizeof(float),cudaMemcpyHostToDevice);

  int aux=length;
  while(aux>1)
  {
     dim3 dimBlock(BLOCK_SIZE,1,1);
     int grid=ceil(aux/(float)BLOCK_SIZE);
      dim3 dimGrid(grid,1,1);
     reduceKernel<<<dimGrid,dimBlock>>>(d_A,d_B,aux);
     cudaDeviceSynchronize();
     cudaMemcpy(d_A,d_B,length*sizeof(float),cudaMemcpyDeviceToDevice);
     aux=(ceil(aux/(float)BLOCK_SIZE));
  }

  cudaMemcpy(B,d_B,length*sizeof(float),cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
}

//======= MAIN function ========================================================
int main ()
{
 int l = 1096000; //Vector's length
 clock_t start, finish;
 double elapsedSecuential, elapsedParallel, optimization;
 float *A = (float *) malloc(l * sizeof(float));
 float *B = (float *) malloc(l * sizeof(float));

 fillVector(A,1.0,l);
 fillVector(B,0.0,l);

 start = clock();
 float sum = serialVectorItemsAdd(A,l);
 finish = clock();
 cout<< "The result is: " << sum << endl;
 elapsedSecuential = (((double) (finish - start)) / CLOCKS_PER_SEC );
 cout<< "The Secuential process took: " << elapsedSecuential << " seconds to execute "<< endl<< endl;

 start = clock();
 vectorItemsAdd(A,B,l);
 finish = clock();
 cout<< "The result is: " << B[0] << endl;
 elapsedParallel = (((double) (finish - start)) / CLOCKS_PER_SEC );
 cout<< "The Parallel process took: " << elapsedParallel << " seconds to execute "<< endl<< endl;

 optimization = elapsedSecuential/elapsedParallel;
 cout<< "The acceleration we've got: " << optimization <<endl;

 cout<< "============================================ "<<endl;
 resultCompare(sum, B);

 free(A);
 free(B);
}
