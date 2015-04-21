#include<iostream>
#include<cstdlib>
#include<cstdlib>
#include<cuda.h>
#include<highgui.h>
#include<cv.h>

#define Mask_size  3
#define TILE_SIZE  1024
#define BLOCK_SIZE 32
__constant__ char M[Mask_size*Mask_size];
//#define clamp(x) (min(max((x), 0.0), 1.0))

using namespace std;
using namespace cv;


__device__ unsigned char delimit(int value)//__device__ because it's called by a kernel
{
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return  value;
}



__global__ void convolution2DGlobalMemKernel(unsigned char *In,char *M, unsigned char *Out,int Mask_Width,int Rowimg,int Colimg)
{

   unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
   unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

   int Pvalue = 0;

   int N_start_point_row = row - (Mask_Width/2);
   int N_start_point_col = col - (Mask_Width/2);

   for(int i = 0; i < Mask_Width; i++)
   {
       for(int j = 0; j < Mask_Width; j++ )
       {
        if((N_start_point_col + j >=0 && N_start_point_col + j < Rowimg)&&(N_start_point_row + i >=0 && N_start_point_row + i < Colimg))
        {
          Pvalue += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * M[i*Mask_Width+j];
        }
       }
   }

   Out[row*Rowimg+col] = delimit(Pvalue);

}

__global__ void convolution2DConstantMemKernel(unsigned char *In,unsigned char *Out,int Mask_Width,int Rowimg,int Colimg)
 {
   unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
   unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

   int Pvalue = 0;

   int N_start_point_row = row - (Mask_Width/2);
   int N_start_point_col = col - (Mask_Width/2);

   for(int i = 0; i < Mask_Width; i++)
   {
       for(int j = 0; j < Mask_Width; j++ )
       {
         if((N_start_point_col + j >=0 && N_start_point_col + j < Rowimg)&&(N_start_point_row + i >=0 && N_start_point_row + i < Colimg))
         {
           Pvalue += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * M[i*Mask_Width+j];
         }
       }
    }

   Out[row*Rowimg+col] = delimit(Pvalue);
}

void convolution2DKernelCall(Mat image,unsigned char *In,unsigned char *Out,char *h_Mask,
  int Mask_Width,int Row,int Col, int op)
{
  // Variables
  int Size_of_bytes =  sizeof(unsigned char)*Row*Col*image.channels();
  int Mask_size_bytes =  sizeof(char)*(Mask_size*Mask_size);
  unsigned char *d_In, *d_Out;
  char *d_Mask;
  float Blocksize=BLOCK_SIZE;


  // Memory Allocation in device
  cudaMalloc((void**)&d_In,Size_of_bytes);
  cudaMalloc((void**)&d_Out,Size_of_bytes);
  cudaMalloc((void**)&d_Mask,Mask_size_bytes);
  // Memcpy Host to device
  cudaMemcpy(d_In,In,Size_of_bytes,cudaMemcpyHostToDevice);
  cudaMemcpy(d_Mask,h_Mask,Mask_size_bytes,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(M,h_Mask,Mask_size_bytes);// Using constant mem

  dim3 dimGrid(ceil(Row/Blocksize),ceil(Col/Blocksize),1);
  dim3 dimBlock(Blocksize,Blocksize,1);
  switch(op)
  {
    case 1:
    cout<<"2D convolution using GLOBAL mem"<<endl;
    convolution2DGlobalMemKernel<<<dimGrid,dimBlock>>>(d_In,d_Mask,d_Out,Mask_Width,Row,Col);
    break;
    case 2:
    cout<<"2D convolution using CONSTANT mem"<<endl;
    convolution2DConstantMemKernel<<<dimGrid,dimBlock>>>(d_In,d_Out,Mask_Width,Row,Col);
    break;
    case 3:
    cout<<"2D convolution using SHARED mem... Soon"<<endl;
  }

  cudaDeviceSynchronize();
  // save output result.
  cudaMemcpy (Out,d_Out,Size_of_bytes,cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_In);
  cudaFree(d_Out);
  cudaFree(d_Mask);
}



int main()
{

  clock_t start, finish; //Clock variables
  double elapsedParallel;
  double elapsedSecuential;
  int Mask_Width =  Mask_size;
  Mat image;
  image = imread("inputs/img1.jpg",0);   // Read the file, 0 means we already load de image in gray scale
  Size s = image.size();
  int Row = s.width;
  int Col = s.height;
  char h_Mask[] = {-1,0,1,-2,0,2,-1,0,1}; // A kernel for edge detection
  //another mask option could be {-1,-2,-1,0,0,0,1,2,1} if you want to use this filter in the Y axis

  unsigned char *img = (unsigned char*)malloc(sizeof(unsigned char)*Row*Col*image.channels());
  unsigned char *imgOut = (unsigned char*)malloc(sizeof(unsigned char)*Row*Col*image.channels());

  img = image.data;


  cout<<"Parallel result"<<endl;
  start = clock();
  convolution2DKernelCall(image,img,imgOut,h_Mask,Mask_Width,Row,Col,1);
  finish = clock();
  elapsedParallel = (((double) (finish - start)) / CLOCKS_PER_SEC );
  cout<< "The parallel process took: " << elapsedParallel << " seconds to execute "<< endl;

  cout<<"Serial result"<<endl;
  Mat grad_x, abs_grad_x;
  start = clock();
  Sobel(image,grad_x,CV_8UC1,1,0,3,1,0,BORDER_DEFAULT);
  convertScaleAbs(grad_x, abs_grad_x);
  finish = clock();
  elapsedSecuential = (((double) (finish - start)) / CLOCKS_PER_SEC );
  cout<< "The Secuential process took: " << elapsedSecuential << " seconds to execute "<< endl;


  Mat gray_image;
  gray_image.create(Row,Col,CV_8UC1);
  gray_image.data = imgOut;
  imwrite("./outputs/1053823121.png",gray_image);
  //Wilson if youŕe gonna use this code change the name of the image for your code

  //free(img);
  //free(imgOut);

  return 0;
}
