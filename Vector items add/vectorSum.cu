  #include<cstdlib>
  #include<time.h>
  #include<cuda.h>
  #include<iostream>

  using namespace std;

  int serialVectorItemsAdd (int *A, int size)
  {
    int sum=0;

    for (int i = 0; i < size; i++)
    {
      sum = sum + A[i];
    }
    return sum;
  }

  void fillVector (int *A, int value, int size )
  {
    for (int i=0; i<size; i++)
    {
      A[i] = value;
    }
  }

  //Parallel
  __global__ void reduceKernel(int *g_idata, int *g_odata)
  {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
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
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
  }

  void vectorItemsAdd(int *A, int size)
  {

  }


 int main ()
 {
   int l = 5;
   int *A = (int *) malloc(l * sizeof(int));

   fillVector(A,2,l);
   int sum = serialVectorItemsAdd(A,l);
   cout<< "the result is: " << sum << endl;
  free(A);
  }
