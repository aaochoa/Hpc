#include<iostream>
#include<cstdlib>
#include<cstdlib>
#include<cuda.h>
#include<highgui.h>
#include<cv.h>

#define Mask_size  3
#define TILE_SIZE  32
#define BLOCK_SIZE 32
__constant__ char M[Mask_size*Mask_size];
//#define clamp(x) (min(max((x), 0.0), 1.0))

using namespace std;
using namespace cv;


//============ TO HOLD THE VALUES AND DON LET THEM GET OUT OF THE DOMAIN =======

__device__ unsigned char clamp(int value)//__device__ because it's called by a kernel
{
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return  value;
}


//============= CONVOLUTION KERNEL WITH GLOBAL MEM ============================

__global__ void convolution2DGlobalMemKernel(unsigned char *In,char *Mask, unsigned char *Out,int Mask_Width,int Rowimg,int Colimg)
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
        if((N_start_point_col + j >=0 && N_start_point_col + j < Rowimg)
        &&(N_start_point_row + i >=0 && N_start_point_row + i < Colimg))
        {
          Pvalue += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * Mask[i*Mask_Width+j];
        }
       }
   }

   Out[row*Rowimg+col] = clamp(Pvalue);

}

//============== CONVOLUTION KERNEL WITH CONSTANT MEM =========================

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
         if((N_start_point_col + j >=0 && N_start_point_col + j < Rowimg)
         &&(N_start_point_row + i >=0 && N_start_point_row + i < Colimg))
         {
           Pvalue += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * M[i*Mask_Width+j];
         }
       }
    }

   Out[row*Rowimg+col] = clamp(Pvalue);
}

//============== CONVOLUTION KERNEL WITH SHARED MEM =========================

__global__ void convolution2DSharedMemKernel(unsigned char *imageInput,unsigned char *imageOutput,
 int maskWidth, int width, int height)
{
    __shared__ float N_ds[TILE_SIZE + Mask_size - 1][TILE_SIZE + Mask_size - 1];
    int n = maskWidth/2;
    int dest = threadIdx.y*TILE_SIZE+threadIdx.x, destY = dest / (TILE_SIZE+Mask_size-1), destX = dest % (TILE_SIZE+Mask_size-1),
        srcY = blockIdx.y * TILE_SIZE + destY - n, srcX = blockIdx.x * TILE_SIZE + destX - n,
        src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        N_ds[destY][destX] = imageInput[src];
    else
        N_ds[destY][destX] = 0;

    // Second batch loading
    dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
    destY = dest /(TILE_SIZE + Mask_size - 1), destX = dest % (TILE_SIZE + Mask_size - 1);
    srcY = blockIdx.y * TILE_SIZE + destY - n;
    srcX = blockIdx.x * TILE_SIZE + destX - n;
    src = (srcY * width + srcX);
    if (destY < TILE_SIZE + Mask_size - 1)
    {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = imageInput[src];
        else
            N_ds[destY][destX] = 0;
    }
    __syncthreads();

    int Pvalue = 0;
    int y, x;
    for (y = 0; y < maskWidth; y++)
        for (x = 0; x < maskWidth; x++)
            Pvalue += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * maskWidth + x];
    y = blockIdx.y * TILE_SIZE + threadIdx.y;
    x = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (y < height && x < width)
        imageOutput[(y * width + x)] = clamp(Pvalue);
    __syncthreads();
}

//============ KERNEL CALL =====================================================

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
  switch(op)//to select which kernel we want to execute alongside the Secuential version
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
    cout<<"2D convolution using SHARED mem"<<endl;
    convolution2DSharedMemKernel<<<dimGrid,dimBlock>>>(d_In,d_Out,Mask_Width,Row,Col);
    break;
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
  double elapsedSequential;
  int Mask_Width =  Mask_size;
  Mat image;
  image = imread("inputs/img4.jpg",0);   // Read the file, 0 means we already load de image in gray scale
  int op = 3;
  Size s = image.size();
  int Row = s.width;
  int Col = s.height;
  char h_Mask[] = {-1,0,1,-2,0,2,-1,0,1}; // A kernel for edge detection
  //another mask option could be {-1,-2,-1,0,0,0,1,2,1} if you want to use this filter in the Y axis

  //image.channels() don't needed because the image is already in gray scale
  unsigned char *img = (unsigned char*)malloc(sizeof(unsigned char)*Row*Col*image.channels());
  unsigned char *imgOut = (unsigned char*)malloc(sizeof(unsigned char)*Row*Col*image.channels());

  if( !image.data )
  {//To test out if the image was loaded properly
    cout<<"Problems loading the image"<<endl;
    return -1;
  }

  img = image.data;

  cout<<"Serial result"<<endl;
  Mat grad_x;
  start = clock();
  Sobel(image,grad_x,CV_8UC1,1,0,3,1,0,BORDER_DEFAULT);
  finish = clock();
  //imwrite("./outputs/1053823121.png",grad_x);
  elapsedSequential = (((double) (finish - start)) / CLOCKS_PER_SEC );
  cout<< "The Secuential process took: " << elapsedSequential << " seconds to execute "<< endl;

  start = clock();
  convolution2DKernelCall(image,img,imgOut,h_Mask,Mask_Width,Row,Col,op);
  finish = clock();
  elapsedParallel = (((double) (finish - start)) / CLOCKS_PER_SEC );
  cout<< "The parallel process took: " << elapsedParallel << " seconds to execute "<< endl;

  Mat gray_image;
  gray_image.create(Col,Row,CV_8UC1);
  gray_image.data = imgOut;
  imwrite("./outputs/1053823121.png",gray_image);
  //Wilson if youŕe gonna use this code change the name of the image for your code

  return 0;
}
