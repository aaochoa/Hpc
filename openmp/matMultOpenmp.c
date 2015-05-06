//Anderson Alberto Ochoa Estupiñan
//Code: 1053823121

#include<stdio.h>
#include<iostream>
#include<cstdlib>
#include<time.h>
#include<omp.h>

using namespace std;

//=====================================================================================
//Function to print matrices
void print(int *A, int n, int m)
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
void fillMatrix (int *mat, int value, int n, int m)
{
  int size=n*m;

  for (int i=0; i<size; i++)
  {
    mat[i] = value;
  }
}

//=====================================================================================
//Function used to compare the results
int compareMatrix (int *A, int *B,int n, int m)
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

//=====================================================================================
//Sequential
//Function used to multiply both matrices taking each matrix as a vector
void multMatrixsequential (int *h_matA, int *h_matB, int *h_matC, int n, int m, int o)
{
  //Row*Width+Col to find the value in the given bidimensional index
  for (int i=0; i<n; i++)
  {
    for (int j=0; j<o; j++)
    {
      int sum=0;
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

//====================================================================================
int main()
{
    clock_t start, finish;
    double elapsedsequential,elapsedParallel;
    int n=500;
    int m=480;
    int o=1000;

    int i,j,k;
    int chunk = 10;

    int *matA = (int *) malloc(n * m * sizeof(int));
    int *matB = (int *) malloc(m * o * sizeof(int));
    int *matCS = (int *) malloc(n * o * sizeof(int));
    int *matCP = (int *) malloc(n * o * sizeof(int));

    fillMatrix(matA,1,n,m);
    fillMatrix(matB,1,m,o);
    fillMatrix(matCS,0,n,o);
    fillMatrix(matCP,0,n,o);

    start = clock();
    multMatrixsequential(matA,matB,matCS,n,m,o);
    finish = clock();
    elapsedsequential = (((double) (finish - start)) / CLOCKS_PER_SEC );
    cout<< "The Sequential process took: " << elapsedsequential << " seconds to execute "<< endl<< endl;

    start = clock();
    #pragma omp parallel shared(matA,matB,matCP) private (i,j,k)
    {
      #pragma omp for schedule (static,chunk)
      for (i=0; i<n; i++)
      {
        for (j=0; j<o; j++)
        {
          int sum=0;
          for (k=0; k<m; k++)
          {
            sum += matA[m*i+k]*matB[o*k+j];
          }
          matCP[o*i+j] = sum;
        }
      }
    }
    finish = clock();
    elapsedParallel = (((double) (finish - start)) / CLOCKS_PER_SEC );
    cout<< "The Parallel process took: " << elapsedParallel << " seconds to execute "<< endl<< endl;

    cout<< "Comparing Serial vs Parallel result " <<endl;
    compareMatrix(matCS,matCP,n,o);

    //For debugging porpouses only
    //print(matCS,n,o);
    //cout<<endl;
    //print(matCP,n,o);
    //cout<<endl;

    free(matA);
    free(matB);
    free(matCS);
    free(matCP);
}
