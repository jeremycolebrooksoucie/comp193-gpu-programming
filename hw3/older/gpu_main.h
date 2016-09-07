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
    int gDIM;       // size of image (square)

    unsigned long gSize;      // size of vector of data for image
    float* gray;              // grayscale data
    float* red; 
    float* green; 
    float* blue; 
        
    curandState* rand;

};

int updatePalette(GPU_Palette* P, const float* inPtr);
GPU_Palette initGPUPalette(AParams* PARAMS);
int freeGPUPalette(GPU_Palette* P1);

// kernel calls:
__global__ void updateReds(float* red, curandState* rand, float amp);
__global__ void updateGreens(float* green, curandState* rand, float amp);
__global__ void updateBlues(float* blue, curandState* rand, float amp);


#endif  // GPULib

