//Matrix multiplication using floats
//Anderson Alberto Ochoa Estupiñan
//Code: 1053823121



#include<stdio.h>
#include<iostream>
#include<cstdlib>
#include<time.h>
#include<cuda.h>

#define TILE_WIDTH 32
using namespace std;

//=====================================================================================
//Function to print matrices
void print(float *A, int n, int m)
{
    for (int i=0; i<n; i++)
    {
      for (int j=0; j<m; j++)
      {
        cout<<A[n*i+j]<<" | ";
      }
      cout<<endl;
    }
}

//=====================================================================================
//Function used just to fill the given matrix with a given value
void fillMatrix (float *mat, float value, int n, int m)
{
  int size=n*m;

  for (int i=0; i<size; i++)
  {
    mat[i] = value;
  }
}

//=====================================================================================
//sequential
//Function used to multiply both matrices taking each matrix as a vector
void multMatrixsequential (float *h_matA, float *h_matB, float *h_matC, int n, int m, int o)
{
  //Row*Width+Col to find the value in the given bidimensional index
  for (int i=0; i<n; i++)
  {
    for (int j=0; j<o; j++)
    {
      float sum=0;
      for (int k=0; k<m; k++)
      {
        sum += h_matA[m*i+k]*h_matB[o*k+j];
      }
      h_matC[o*i+j] = sum;
      //cout<<h_matC[n*i+j]<<" | ";
    }
    //cout<<endl;
  }
}

//=====================================================================================
//Parallel
//The multiplication kernel without tiles
__global__ void matrixMultKernel (float *d_matA, float *d_matB, float *d_matC, int n, int m, int o)
{
  int Row = blockIdx.y*blockDim.y+threadIdx.y;
  int Col = blockIdx.x*blockDim.x+threadIdx.x;

  if ((Row<n)&&(Col<o))
  {
    float temp=0;

    for (int i=0; i<m; i++)
    {
      temp += d_matA[Row*m+i]*d_matB[i*o+Col];
    }
    d_matC[Row*o+Col] = temp;
  }
}

//=====================================================================================
//the multiplication kernel with tiles
__global__ void matrixMulKernelTiled(float *d_matA, float *d_matB, float *d_matC, int n, int m, int o){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for (int k = 0; k < (m+TILE_WIDTH-1)/(TILE_WIDTH); ++k)
    {
      if (k*TILE_WIDTH + tx < m && row < n)
      {
          Mds[ty][tx] = d_matA[row * m + k*TILE_WIDTH + tx];
      } else
      {
        Mds[ty][tx] = 0;
      }

      if (k*TILE_WIDTH + ty < m && col < o)
      {
          Nds[ty][tx] = d_matB[(k*TILE_WIDTH + ty) * o + col];
      } else
      {
        Nds[ty][tx] =0;
      }

        __syncthreads();
      for(int k = 0; k < TILE_WIDTH; ++k)
      {
        Pvalue += Mds[ty][k] * Nds[k][tx];
      }
      __syncthreads();
  }

  if (row < n && col < o)
  {
    d_matC[row * o + col] = Pvalue;
  }

}


//=====================================================================================
//Function to call the kernel of the tiled multiplication
void multMatrixParallelTiled(float *A, float *B, float *C, int n, int m, int o)
{

  float blockSize = TILE_WIDTH;

  float *d_matA, *d_matB, *d_matC;

  //1. Allocate memory for d_matA, etc. on the device (cudaMalloc)
  cudaMalloc(&d_matA, n * m * sizeof(float));
  cudaMalloc(&d_matB, m * o * sizeof(float));
  cudaMalloc(&d_matC, n * o * sizeof(float));
  //2. Copy Data from host to d_matA, etc. (cudaMemcpy)
  cudaMemcpy(d_matA, A, n * m * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_matB, B, m * o * sizeof(float), cudaMemcpyHostToDevice);
  dim3 threads(blockSize,blockSize,1); //How many blocks U want in each direction -- U have to respect the GPU's capacity
  dim3 blocks(ceil(o/blockSize),ceil(n/blockSize),1);//How many threads U want to have per block --
  //The GPU used in this course is capable of have 1024 threads per block
  //3. Kernel Launch Code
  matrixMultKernel<<<blocks,threads>>>(d_matA,d_matB,d_matC,n,m,o);
  cudaMemcpy (C, d_matC, n * o * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_matA);
  cudaFree(d_matB);
  cudaFree(d_matC);

}

//=====================================================================================
//Function to call the tile less multiplication kernel
void multMatrixParallel(float *A, float *B, float *C, int n, int m, int o)
{

    float blockSize = TILE_WIDTH;

    float *d_matA, *d_matB, *d_matC;

    //1. Allocate memory for d_matA, etc. on the device (cudaMalloc)
    cudaMalloc(&d_matA, n * m * sizeof(float));
    cudaMalloc(&d_matB, m * o * sizeof(float));
    cudaMalloc(&d_matC, n * o * sizeof(float));
    //2. Copy Data from host to d_matA, etc. (cudaMemcpy)
    cudaMemcpy(d_matA, A, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, B, m * o * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threads(blockSize,blockSize,1); //How many blocks U want in each direction -- U have to respect the GPU's capacity
    dim3 blocks(ceil(o/blockSize),ceil(n/blockSize),1);//How many threads U want to have per block --
    //The GPU used in this course is capable of have 1024 threads per block
    //3. Kernel Launch Code
    matrixMultKernel<<<blocks,threads>>>(d_matA,d_matB,d_matC,n,m,o);
    cudaMemcpy (C, d_matC, n * o * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);
}


//=====================================================================================
//Function used to compare the results
int compareMatrix (float *A, float *B,int n, int m)
{
  int size=n*m;
  for (int i=0; i<size; i++ )
  {
    if (A[i]!=B[i])
    {
      cout<<"## sequential and Parallel results are NOT equal ##"<<endl;
      return 0;
    }
  }
  cout<<"== sequential and Parallel results are equal =="<<endl;
  return 0;
}


//========================================== MAIN =====================================

int main()
{
    clock_t start, finish;
    double elapsedsequential,elapsedParallel,elapsedParallelTiles,optimizationP,optimizationT;
    int n=2;
    int m=4;
    int o=8;

    float *matA = (float *) malloc(n * m * sizeof(float));
    float *matB = (float *) malloc(m * o * sizeof(float));
    float *matCS = (float *) malloc(n * o * sizeof(float));
    float *matCP = (float *) malloc(n * o * sizeof(float));
    float *matCPT = (float *) malloc(n * o * sizeof(float));

    fillMatrix(matA,1.5,n,m);
    fillMatrix(matB,1.5,m,o);
    fillMatrix(matCS,0,n,o);
    fillMatrix(matCP,0,n,o);
    fillMatrix(matCPT,0,n,o);

    start = clock();
    multMatrixsequential(matA,matB,matCS,n,m,o);
    finish = clock();
    elapsedsequential = (((double) (finish - start)) / CLOCKS_PER_SEC );
    cout<< "The sequential process took: " << elapsedsequential << " seconds to execute "<< endl<< endl;

    start = clock();
    multMatrixParallel(matA,matB,matCP,n,m,o);
    finish = clock();
    elapsedParallel = (((double) (finish - start)) / CLOCKS_PER_SEC );
    cout<< "The parallel process took: " << elapsedParallel << " seconds to execute "<< endl<< endl;

    start = clock();
    multMatrixParallelTiled(matA,matB,matCPT,n,m,o);
    finish = clock();
    elapsedParallelTiles = (((double) (finish - start)) / CLOCKS_PER_SEC );
    cout<< "The parallel process using Tiles took: " << elapsedParallelTiles << " seconds to execute "<< endl<< endl;

    optimizationP = elapsedsequential/elapsedParallel;
    cout<< "The acceleration we've got without using Tiles: " << optimizationP << "X" <<endl;

    optimizationT = elapsedsequential/elapsedParallelTiles;
    cout<< "The acceleration we've got using Tiles: " << optimizationT << "X" <<endl;

    cout<< "Comparing Serial vs Parallel result " <<endl;
    compareMatrix(matCS,matCP,n,o);
    cout<< "Comparing Serial vs Parallel with Tiles result " <<endl;
    compareMatrix(matCS,matCPT,n,o);

    //For debugging porpouses only
    //print(matCS,n,o);
    //cout<<endl;
    //print(matCP,n,o);
    //cout<<endl;
    //print(matCPT,n,o);

    free (matA);
    free (matB);
    free (matCS);
    free (matCP);
    free (matCPT);
    return 0;
}
