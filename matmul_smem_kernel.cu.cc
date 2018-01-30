// matmul_smem_kernel.cu.cc
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <stdio.h>
//#include "matmul_smem_kernel.h"  //(for some reason some third party inclusions  (eigen, cuda) fail when in a
// header file, which why I declare the kernel launcher here for now.)

using GPUDevice = Eigen::GpuDevice;


template <typename GPUDevice, typename T>
struct MatMulSharedMemKernelLauncher {
  void operator() (const GPUDevice& d, const T * A, const T * B, T * C,
                   const int numARows, const int numACols,
                   const int numBRows, const int numBCols);
};


#define TILE_WIDTH 64 //on a gtx 1070, we have up to 49,152 bytes of shared mem
// per block, so here 64x64 integers = 64x64x8 = 32,768 bytes,  will fit.

template <typename T>
__global__ void MatMulSharedMemKernel(const T * A, const T * B, T * C, const int numARows, const int numACols,
                                      const int numBRows, const int numBCols) {

  __shared__ T ds_Asub[TILE_WIDTH][TILE_WIDTH];
  __shared__ T ds_Bsub[TILE_WIDTH][TILE_WIDTH];

  int bx=blockIdx.x, by=blockIdx.y,
    tx=threadIdx.x, ty=threadIdx.y,
    row=by*TILE_WIDTH + ty,
    col=bx*TILE_WIDTH + tx;
  
  T Pvalue=0;

  for(int m=0; m<(numACols-1)/TILE_WIDTH + 1; ++m){
    
    if(row<numARows && m*TILE_WIDTH+tx < numACols){
      ds_Asub[ty][tx]=A[row*numACols + m*TILE_WIDTH + tx];
    }else{
      ds_Asub[ty][tx]=0;
    }
    if((m*TILE_WIDTH+ty) < numBRows && col<numBCols){
      ds_Bsub[ty][tx]=B[(m*TILE_WIDTH+ty)*numBCols + col];
    }else{
      ds_Bsub[ty][tx]=0;
    }
    
    __syncthreads();

    for(int k=0; k<TILE_WIDTH; ++k){
      Pvalue += ds_Asub[ty][k]*ds_Bsub[k][tx];
    }
    
    __syncthreads();
        
  }
  
  if(row < numARows && col < numBCols){
    C[row*numBCols + col] = Pvalue;
   }
}

template <typename GPUDevice, typename T>
void MatMulSharedMemKernelLauncher<GPUDevice, T>::operator()(const GPUDevice& d, const T * A, const T * B, T * C,
                                                             const int numARows, const int numACols,
                                                             const int numBRows, const int numBCols) {

  dim3 dimGrid((numBCols-1)/TILE_WIDTH + 1, (numARows-1)/TILE_WIDTH +1,1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  
  MatMulSharedMemKernel<T><<<dimGrid, dimBlock, 0, d.stream()>>>(A, B, C, numARows, numACols, numBRows, numBCols);
}

template struct MatMulSharedMemKernelLauncher<GPUDevice, int>;
template struct MatMulSharedMemKernelLauncher<GPUDevice, float>;

#endif
