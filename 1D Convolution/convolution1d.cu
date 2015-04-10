//============================================================================
#include<cstdlib>
#include<time.h>
#include<cuda.h>
#include<iostream>
#include<math.h> //Included just to use the Power function

using namespace std;

#define MAX_MASK_WIDTH 10
__constant__ float M[MAX_MASK_WIDTH];
cudaMemcpyToSymbol(M, h_M, Mask_Width*sizeof(float));

//====== Function made to print vector =========================================
void printVector (double *A, int length)
{
  for (int i=0; i<length; i++)
  {
    cout<<A[i]<<" | ";
  }
  cout<<endl;
}

//====== Function made to fill the vector with some given value ================
void fillVector (double *A, double value, int length)
{
  for (int i=0; i<length; i++)
  {
    A[i] = value;
  }
}

//====== Basic convolution kernel ==============================================
__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P,
 int Mask_Width, int Width)
 {
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   float Pvalue = 0;
   int N_start_point = i - (Mask_Width/2);
   for (int j = 0; j < Mask_Width; j++)
   {
     if (N_start_point + j >= 0 && N_start_point + j < Width)
     {
       Pvalue += N[N_start_point + j]*M[j];
     }
   }
   P[i] = Pvalue;
}

//====== Convolution kernel using constant memory and caching ==================
//The main diference is that we don't need here to pass M to the function because
//it's now in constant memory thanks to this lines
//__constant__ float M[MAX_MASK_WIDTH];
//cudaMemcpyToSymbol(M, h_M, Mask_Width*sizeof(float));
//because the CUDA runtime knows that constant memory variables are not
//modified during kernel execution, it directs the hardware to aggressively cache the
//constant memory variables during kernel execution

global__ void convolution_1D_basic_kernel(float *N, float *P, int Mask_Width,
 int Width)
 {
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   float Pvalue = 0;
   int N_start_point = i - (Mask_Width/2);
   for (int j = 0; j < Mask_Width; j++)
   {
     if (N_start_point + j >= 0 && N_start_point + j < Width)
     {
       Pvalue += N[N_start_point + j]*M[j];
     }
   }
   P[i] = Pvalue;
}

//===== Tiled Convolution kernel using shared memory ===========================
global__ void convolution_1D_basic_kernel(float *N, float *P, int Mask_Width,
 int Width)
 {
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   __shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1];
   int n = Mask_Width/2;
   int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
   if (threadIdx.x >= blockDim.x - n)
   {
     N_ds[threadIdx.x - (blockDim.x - n)] =
     (halo_index_left < 0) ? 0 : N[halo_index_left];
   }
   N_ds[n + threadIdx.x] = N[blockIdx.x*blockDim.x + threadIdx.x];
   int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
   if (threadIdx.x < n)
   {
     N_ds[n + blockDim.x + threadIdx.x] =
     (halo_index_right >= Width) ? 0 : N[halo_index_right];
   }
   __syncthreads();
   float Pvalue = 0;
   for(int j = 0; j < Mask_Width; j++)
   {
     Pvalue += N_ds[threadIdx.x + j]*M[j];
   }
   P[i] = Pvalue;
}

//====== A simplier tiled convolution kernel using shared memory and general cahching
global__ void convolution_1D_basic_kernel(float *N, float *P, int Mask_Width,
 int Width)
 {
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   __shared__ float N_ds[TILE_SIZE];
   N_ds[threadIdx.x] = N[i];
   __syncthreads();
   int This_tile_start_point = blockIdx.x * blockDim.x;
   int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
   int N_start_point = i - (Mask_Width/2);
   float Pvalue = 0;
   for (int j = 0; j < Mask_Width; j ++)
   {
     int N_index = N_start_point + j;
     if (N_index >= 0 && N_index < Width)
     {
       if ((N_index >= This_tile_start_point)
       && (N_index < Next_tile_start_point))
       {
         Pvalue += N_ds[threadIdx.x+j-(Mask_Width/2)]*M[j];
       } else
       {
         Pvalue += N[N_index] * M[j];
       }
     }
   }
   P[i] = Pvalue;
  }
