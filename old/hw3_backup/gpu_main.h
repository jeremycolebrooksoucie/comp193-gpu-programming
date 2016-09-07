#ifndef GPULib
#define GPULib

#include <cuda.h>
#include <curand.h>                 // includes random num stuff
#include <curand_kernel.h>          // more rand stuff
#include <cuda_texture_types.h>

#include "geometry.h"
#include "params.h"

#define cH2D            cudaMemcpyHostToDevice
#define cD2D            cudaMemcpyDeviceToDevice
#define cD2H            cudaMemcpyDeviceToHost

struct GPU_Palette {

    dim3 gThreads;  // threads (TPB, TPB, 1)
    dim3 gBlocks;   // blocks (DIM/TPB, DIM/TPB, 1)

    int gTPB;       // threads per block
    //int gDIM;       // size of image (square)
    int gWidth;
    int gHeight;

    int print_height, print_width;


    unsigned long gSize;      // size of vector of data for image
    float* gray;              // grayscale data
    float* red;
    float* green;
    float* blue;

    float* gray_background;              // grayscale data
    float* red_background;
    float* green_background;
    float* blue_background;

};

struct GPU_Geometries {
    int size;
    Geometry *geometries;
};


int drawGeometries(GPU_Palette P1, GPU_Geometries G1);
int updateGeometries(GPU_Palette P1, GPU_Geometries G1);


GPU_Palette     initGPUPalette(AParams *PARAMS);
GPU_Geometries  initGPUGeometries(int size, int MEM_SIZE);

int freeGPUPalette(GPU_Palette* P1);

// device helper functions
__device__ void drawCircle(GPU_Palette P1, Geometry g, 
                           int pixelX, int pixelY, int vecIdx);
__device__ Geometry updateCircle(Geometry g, GPU_Palette P1);

__device__ void printGeometry(Geometry g);

// kernel calls:
__global__ void dev_drawGeometries(GPU_Palette P1, GPU_Geometries G1);
__global__ void dev_updateGeometries(GPU_Palette P1, GPU_Geometries G1);


#endif  // GPULib

