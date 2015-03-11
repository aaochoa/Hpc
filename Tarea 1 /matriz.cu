#include<stdio.h>
#include<iostream>
#include<cstdlib>
#include<time.h>
#include<cuda.h>

using namespace std;

void print(int *A, int n)
{
    for (int i=0; i<n; i++)
    {  for (int j=0; j<n; j++)
      {
        cout<<A[n*i+j]<<" | ";
      }
      cout<<endl;
    }
}

//Function used just to fill the given matrix with a given value
void fillMatrix (int *mat, int value, int n)
{
  int size=n*n;

  for (int i=0; i<size; i++)
  {
    mat[i] = value;
  }
}

//Secuential
//Function used to multiply both matrices taking each matrix as a vector
void multMatrixSecuential (int *h_matA, int *h_matB, int *h_matC, int n)
{
  //Row*Width+Col to find the value in the given bidimensional index
  for (int i=0; i<n; i++)
  {
    for (int j=0; j<n; j++)
    { int sum=0;
      for (int k=0; k<n; k++)
      {
        sum += h_matA[n*i+k]*h_matB[n*k+j];
      }
      h_matC[n*i+j] = sum;
      //cout<<h_matC[n*i+j]<<" | ";
    }
    //cout<<endl;
  }

}

//Parallel

__global__ void matrixMultKernel (int *d_matA, int *d_matB, int *d_matC, int n)
{
  int Row = blockIdx.y*blockDim.y+threadIdx.y;
  int Col = blockIdx.x*blockDim.x+threadIdx.x;

  if ((Row<n)&&(Col<n))
  {
    int temp=0;

    for (int i=0; i<n; i++)
    {
      temp += d_matA[Row*n+i]*d_matB[i*n+Col];
    }
    d_matC[Row*n+Col] = temp;
  }
}

void multMatrixParallel(int *A, int *B, int *C, int n)
{
    int size = n * n * sizeof(int);
    float blockSize = 32.0;
    float threadsCount  = n/blockSize;
    clock_t start, finish;
    double elapsedCMC;

    int *d_matA, *d_matB, *d_matC;

    //1. Allocate memory for d_matA, etc. on the device (cudaMalloc)
    cudaMalloc(&d_matA, size);
    cudaMalloc(&d_matB, size);
    cudaMalloc(&d_matC, size);
    //2. Copy Data from host to d_matA, etc. (cudaMemcpy)
    start = clock();
    cudaMemcpy(d_matA, A, size, cudaMemcpyHostToDevice);
    finish = clock();
    elapsedCMC = (((double) (finish - start)) / CLOCKS_PER_SEC );
    cout<< "Matrix A The elapsed time of CudaMemCpy took: " << elapsedCMC << " seconds to execute "<< endl<< endl;
    start = clock();
    cudaMemcpy(d_matB, B, size, cudaMemcpyHostToDevice);
    finish = clock();
    elapsedCMC = (((double) (finish - start)) / CLOCKS_PER_SEC );
    cout<< "Matrix B The elapsed time of CudaMemCpy took: " << elapsedCMC << " seconds to execute "<< endl<< endl;

    dim3 threads(blockSize,blockSize,1); //How many blocks U want in each direction -- U have to respect the GPU's capacity
    dim3 blocks(ceil(threadsCount),ceil(threadsCount),1);//How many threads U want to have per block --
    //The GPU used in this course is capable of have 1024 threads per block

    //3. Kernel Launch Code
    matrixMultKernel<<<blocks,threads>>>(d_matA,d_matB,d_matC,n);
    start = clock();
    cudaMemcpy (C, d_matC, size, cudaMemcpyDeviceToHost);
    finish = clock();
    elapsedCMC = (((double) (finish - start)) / CLOCKS_PER_SEC );
    cout<< "Matrix C The elapsed time of CudaMemCpy took: " << elapsedCMC << " seconds to execute "<< endl<< endl;

    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);
}

void compareMatrix (int *A, int *B,int n)
{
  int size=n*n;
  for (int i=0; i<size; i++ )
  {
    if (A[i]!=B[i])
    {
      cout<<"## Secuential and Parallel results are NOT equal ##"<<endl;
    }
  }
  cout<<"== Secuential and Parallel results are equal =="<<endl;
}

//========================================== MAIN =====================================

int main()
{
    clock_t start, finish;
    double elapsedSecuential,elapsedParallel,optimization;
    int n=1024;
    int size=n * n * sizeof(int);
    int *matA = (int *) malloc(size);
    int *matB = (int *) malloc(size);
    int *matCS = (int *) malloc(size);
    int *matCP = (int *) malloc(size);

    fillMatrix(matA,1,n);
    fillMatrix(matB,1,n);
    fillMatrix(matCS,0,n);
    fillMatrix(matCP,0,n);

    start = clock();

    multMatrixSecuential(matA,matB,matCS,n);

    finish = clock();
    elapsedSecuential = (((double) (finish - start)) / CLOCKS_PER_SEC );
    cout<< "The secuential process took: " << elapsedSecuential << " seconds to execute "<< endl<< endl;

    start = clock();

    multMatrixParallel(matA,matB,matCP,n);

    finish = clock();
    elapsedParallel = (((double) (finish - start)) / CLOCKS_PER_SEC );
    cout<< "The parallel process took: " << elapsedParallel << " seconds to execute "<< endl<< endl;

    optimization = elapsedSecuential/elapsedParallel;
    cout<< "The acceleration we've got: " << optimization <<endl;

    compareMatrix(matCS,matCP,n);

    //For debugging porpouses only
    //print(matCS,n);
    //cout<<endl;
    //print(matCP,n);
    free (matA);
    free (matB);
    free (matCS);
    free (matCP);
    return 0;
}
/*
with n=2000;
The secuential process took: 29.0975 seconds to execute

Matrix A The elapsed time of CudaMemCpy took: 0.002913 seconds to execute

Matrix B The elapsed time of CudaMemCpy took: 0.003077 seconds to execute

Matrix C The elapsed time of CudaMemCpy took: 0.170711 seconds to execute

The parallel process took: 0.273568 seconds to execute

The acceleration we've got: 106.363
== Secuential and Parallel results are equal ==

with n=2048
The secuential process took: 113.699 seconds to execute

Matrix A The elapsed time of CudaMemCpy took: 0.003428 seconds to execute

Matrix B The elapsed time of CudaMemCpy took: 0.003251 seconds to execute

Matrix C The elapsed time of CudaMemCpy took: 0.180522 seconds to execute

The parallel process took: 0.274756 seconds to execute

The acceleration we've got: 413.818
== Secuential and Parallel results are equal ==

*/
