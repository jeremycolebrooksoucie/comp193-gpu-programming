#include <math.h>
#include <cuda.h>
//#include <curand.h>                 // includes random num stuff
//#include <curand_kernel.h>        // more rand stuff
#include <cuda_texture_types.h>

#include <stdio.h>
#include <stdlib.h>
#include "gpu_main.h"
#include "params.h"
#include "geometry.h"

// define texture memory
texture<float, 2> texGray;
texture<float, 2> texRed;
texture<float, 2> texGreen;
texture<float, 2> texBlue;

/*************************************************************************/


int drawGeometries(GPU_Palette P1, GPU_Geometries G1) {
    dev_drawGeometries <<< P1.gBlocks, P1.gThreads >>> (P1, G1);

    return 0;
}

int updateGeometries(GPU_Palette P1, GPU_Geometries G1) {
    dev_updateGeometries <<< P1.gBlocks, P1.gThreads >>> (P1, G1);

    return 0;
}

/*************************************************************************/



__global__ void dev_drawGeometries(GPU_Palette P1, GPU_Geometries G1)
{
    int num_geometries = G1.size;

    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);
    int vecIdx = x + (y * blockDim.x * gridDim.x);

    // reset working pixel to appropriate background
    P1.red[vecIdx] = P1.red_background[vecIdx];
    P1.blue[vecIdx] = P1.blue_background[vecIdx];
    P1.green[vecIdx] = P1.green_background[vecIdx];
    P1.gray[vecIdx] = P1.gray_background[vecIdx];

    for (int i = 0; i < num_geometries; i++) {
        Geometry g = G1.geometries[i];
        if(vecIdx == 0) {
            printf("Printing Geometry %d\n", i);
            printGeometry(g);
        }


        switch(g.type) {
            case CIRCLE :
                drawCircle(P1, g, x, y, vecIdx);
                break; 
        }
    }
}

__global__ void dev_updateGeometries(GPU_Palette P1, GPU_Geometries G1) {
    int num_geometries = G1.size;

    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);
    int vecIdx = x + (y * blockDim.x * gridDim.x);



    if (vecIdx < num_geometries) {
        Geometry g= G1.geometries[vecIdx];
        switch(g.type) {
            case CIRCLE :
                g = updateCircle(g, P1);
                G1.geometries[vecIdx] = g;
                break; 
        }

    }
}


/*************************************************************************/

/*
 * Updates P1's rgb values at pixelX, pixelY to reflect circle in goemetry g
 */
__device__ void drawCircle(GPU_Palette P1, Geometry g, int pixelX, int pixelY, int vecIdx)
{
    Circle c;
    c = g.shape.circle;

    int centerX, centerY;
    centerX = c.center.x;
    centerY = c.center.y;



    double distance;
    distance = sqrtf((pixelX - centerX) * (pixelX - centerX) +
                    (pixelY - centerY) * (pixelY - centerY));

    if (c.fill) {
        if (distance < (c.radius + g.thickness)) {
            (P1.red)[vecIdx] = g.color.r;
            (P1.green)[vecIdx] = g.color.g;
            (P1.blue)[vecIdx] = g.color.b;
        }

    } else {
        if (abs(distance - (double) c.radius) <= g.thickness) {
            (P1.red)[vecIdx] = g.color.r;
            (P1.green)[vecIdx] = g.color.g;
            (P1.blue)[vecIdx] = g.color.b;
        }
    }
}

/*
 *  Updates and returns geometry g based properties 
 *  TODO: code here is kind of unelegant
 */
__device__ Geometry updateCircle(Geometry g, GPU_Palette P1) 
{
    Properties p = g.properties;
    Point center = g.shape.circle.center;
    center.x = center.x + p.momentumX;
    center.y = center.y + p.momentumY;

    if(center.x < 0 || center.x > P1.print_width)
        p.momentumX = -1 * p.momentumX;
    if(center.y < 0 || center.y > P1.print_height)
        p.momentumY = -1 * p.momentumY;
    


    g.shape.circle.center = center;
    g.properties = p;

    return g;

}


__device__ void printGeometry(Geometry g)
{
    switch (g.type) {
        case CIRCLE:
            printf("Cicle\n");
            break;
    }
}


/*************************************************************************/



/*************************************************************************/
GPU_Palette initGPUPalette(AParams* PARAMS) 
{

    // load
    GPU_Palette P;

    P.gTPB = THREADS_PER_BLOCK;      // threads per block
    //P.gDIM = 800;     // assumes the image is 800x800
    P.gWidth = PARAMS->width;
    P.gHeight = PARAMS->height;

    P.print_height = PARAMS -> print_height;
    P.print_width = PARAMS -> print_width;
    // 800x800 palette = 25x25 grid of 32x32 threadblocks
    P.gSize = P.gWidth * P.gHeight * sizeof(float);


    P.gThreads.x = P.gTPB;
    P.gThreads.y = P.gTPB;
    P.gThreads.z = 1;         // 3D of threads allowed
    P.gBlocks.x = (P.gWidth  + P.gTPB - 1) / P.gTPB;
    P.gBlocks.y = (P.gHeight + P.gTPB - 1) / P.gTPB;
    P.gBlocks.z = 1;          // only 2D of blocks allowed

    // allocate memory for the palette
    cudaMalloc((void**) &P.gray, P.gSize);    // black and white (avg of rgb)
    cudaMalloc((void**) &P.red, P.gSize);   // r
    cudaMalloc((void**) &P.green, P.gSize); // g
    cudaMalloc((void**) &P.blue, P.gSize);  // b

    cudaMalloc((void**) &P.gray_background, P.gSize);    // black and white (avg of rgb)
    cudaMalloc((void**) &P.red_background, P.gSize);   // r
    cudaMalloc((void**) &P.green_background, P.gSize); // g
    cudaMalloc((void**) &P.blue_background, P.gSize);  // b

    // create texture memory and bind to black and white data
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    /*
     * Unclear if these should be bound to background memory or display memory TODO
     */ 
    cudaBindTexture2D(NULL, texBlue, P.blue, desc, P.gWidth,
                      P.gHeight, sizeof(float) * P.gWidth);
    cudaBindTexture2D(NULL, texGreen, P.green, desc, P.gWidth,
                      P.gHeight, sizeof(float) * P.gWidth);
    cudaBindTexture2D(NULL, texRed, P.red, desc, P.gWidth,
                      P.gHeight, sizeof(float) * P.gWidth);
    cudaBindTexture2D(NULL, texGray, P.gray, desc, P.gWidth,
                      P.gHeight, sizeof(float) * P.gWidth);
    return P;
}


GPU_Geometries  initGPUGeometries(int size, int mem_size)
{
    GPU_Geometries G;
    G.size = size;
    cudaMalloc((void **) &G.geometries, mem_size);


    //cudaChannelFormatDesc desc = cudaCreateChannelDesc<Geometry>();

    //cudaBindTexture(NULL, texGeometries, G.geometries, mem_size);

    return G;
}

/*************************************************************************/
int freeGPUPalette(GPU_Palette* P) {

    // free texture memory
    cudaUnbindTexture(texGray);
    cudaUnbindTexture(texRed);
    cudaUnbindTexture(texGreen);
    cudaUnbindTexture(texBlue);

    // free gpu memory
    cudaFree(P->gray);
    cudaFree(P->red);
    cudaFree(P->green);
    cudaFree(P->blue);

    return 0;
}

/*************************************************************************/
