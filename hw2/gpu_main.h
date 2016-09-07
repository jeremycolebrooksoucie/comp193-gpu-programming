#ifndef GPULib
#define GPULib

#include <cuda.h>
#include <curand.h>                 // includes random num stuff
#include <curand_kernel.h>          // more rand stuff
#include <cuda_texture_types.h>

#include "params.h"

#define cH2D            cudaMemcpyHostToDevice
#define cD2D            cudaMemcpyDeviceToDevice
#define cD2H            cudaMemcpyDeviceToHost

struct GPU_Palette{
    
    dim3 gThreads;  // threads (TPB, TPB, 1)
    dim3 gBlocks;   // blocks (DIM/TPB, DIM/TPB, 1)
      
    int gTPB;       // threads per block
    //int gDIM;       // size of image (square)
    int gWidth;
    int gHeight;


    unsigned long gSize;      // size of vector of data for image
    float* gray;              // grayscale data
    float* red; 
    float* green; 
    float* blue; 

};

int spreadColor(GPU_Palette* P);
int reduceToEdges(GPU_Palette* P);
int colorAroundEdges(GPU_Palette* P);

GPU_Palette initGPUPalette(AParams* PARAMS);
int freeGPUPalette(GPU_Palette* P1);

// device helper functions
__device__ float partial_diff(char axis, char color, int x, int y);

// kernel calls:

__global__ void dev_reduceToEdges(float* gray);
__global__ void dev_colorAroundEdges(float* red, float* green, float* blue, float* gray);
__global__ void dev_spreadColor(float* red, float* green, float* blue);


#endif  // GPULib

