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


int drawGeometries(GPU_Palette gpuPallete, GPU_Geometries gpuGeometries, GPU_Midi gpuMidi, AParams params) {
    dev_drawGeometries <<< gpuPallete.gBlocks, gpuPallete.gThreads >>> (gpuPallete, gpuGeometries, params);

    return 0;
}

int updateGeometries(GPU_Palette gpuPallete, GPU_Geometries gpuGeometries, GPU_Midi gpuMidi, AParams params) {
    dev_updateGeometries <<< gpuPallete.gBlocks, gpuPallete.gThreads >>> (gpuPallete, gpuGeometries, gpuMidi, params);

    return 0;
}

/*************************************************************************/



__global__ void dev_drawGeometries(GPU_Palette P1, GPU_Geometries G1, AParams params)
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
        //if(vecIdx == 0) {
            //printf("Printing Geometry %d\n", i);
            //printGeometry(g);
        //}

        if (g.displayOn == true) {
            switch(g.type) {
                case CIRCLE :
                    drawCircle(P1, g, x, y, vecIdx, params);
                    break; 
            }
        }
    }
}

__global__ void dev_updateGeometries(GPU_Palette gpuPallete, GPU_Geometries gpuGeometries, GPU_Midi gpuMidi, AParams params) {
    int num_geometries = gpuGeometries.size;

    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);
    int vecIdx = x + (y * blockDim.x * gridDim.x);



    if (vecIdx < num_geometries) {
        int soundIndex = gpuMidi.trackLength * vecIdx + params.curFrame;
        SoundEvent curSound = gpuMidi.tracks[soundIndex];
        Geometry g= gpuGeometries.geometries[vecIdx];
        switch(g.type) {
            case CIRCLE :
                g = updateCircle(g, gpuPallete, curSound);
                gpuGeometries.geometries[vecIdx] = g;
                break; 
        }

    }
}


/*************************************************************************/

/*
 * Updates P1's rgb values at pixelX, pixelY to reflect circle in goemetry g
 */
__device__ void drawCircle(GPU_Palette P1, Geometry g, int pixelX, int pixelY, int vecIdx, AParams params)
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
            if (params.afterimage) {
                (P1.red_background)[vecIdx] = ((P1.red_background)[vecIdx] + g.color.r) / 2;
                (P1.green_background)[vecIdx] = ((P1.green_background)[vecIdx] + g.color.g) / 2;
                (P1.blue_background)[vecIdx] = ((P1.blue_background)[vecIdx] + g.color.b) / 2;
            } else {
                (P1.red)[vecIdx] = g.color.r;
                (P1.green)[vecIdx] = g.color.g;
                (P1.blue)[vecIdx] = g.color.b;
            }


        }

    } else {
        if (abs(distance - (double) c.radius) <= g.thickness) {
            if (params.afterimage) {
                (P1.red_background)[vecIdx] = ((P1.red_background)[vecIdx] + g.color.r) / 2;
                (P1.green_background)[vecIdx] = ((P1.green_background)[vecIdx] + g.color.g) / 2;
                (P1.blue_background)[vecIdx] = ((P1.blue_background)[vecIdx] + g.color.b) / 2;
            } else {
                (P1.red)[vecIdx] = g.color.r;
                (P1.green)[vecIdx] = g.color.g;
                (P1.blue)[vecIdx] = g.color.b;
            }
        }
    }
}

/*
 *  Updates and returns geometry g based properties 
 *  TODO: code here is kind of unelegant
 */
__device__ Geometry updateCircle(Geometry g, GPU_Palette P1, SoundEvent currentSound) 
{
    Properties p;
    Circle c;

    if (currentSound.pitch < 20 || currentSound.volume == 0){
        c.radius = 0;
        g.shape.circle = c;
        g.properties = p;
        g.displayOn = false;
        p.lastSoundEvent = currentSound;

        return g;
    }

    c = g.shape.circle;
    p = g.properties;

    // update position
    c.center.x = c.center.x + p.momentumX;
    c.center.y = c.center.y + p.momentumY;
    if(c.center.x < 0 || c.center.x > P1.print_width)
        p.momentumX = -1 * p.momentumX;
    if(c.center.y < 0 || c.center.y > P1.print_height)
        p.momentumY = -1 * p.momentumY;

    //update color
    if (currentSound.pitch > 20 && currentSound.pitch < 110) {
        int scaledPitch = ( (double) currentSound.pitch - 20.0)/90.0 * 256 * 256 * 256;
        int rRaw, gRaw, bRaw;
        rRaw = (scaledPitch % 256);
        gRaw = ((scaledPitch / 256) % 256);
        bRaw = ((scaledPitch / (256 * 256)) % 256);
        g.color.r = ((double) rRaw) / 256.0;
        g.color.g = ((double) gRaw) / 256.0;
        g.color.b = ((double) bRaw) / 256.0;
        g.displayOn = true;
    }

    // update radius
    if (g.properties.lastSoundEvent.volume == currentSound.volume && g.properties.lastSoundEvent.pitch == currentSound.pitch) {
        p.timeSinceLastChanged++;
    } else {
        // volume of 200 should have radius f half the screen
        int VolumeToRadiusScalingFactor = ((P1.print_width + P1.print_height) /(2 * 2))  / 200; 
        int oldRadiusBound = p.lastSoundEvent.volume * VolumeToRadiusScalingFactor;
        int newRadiusBound = currentSound.volume                * VolumeToRadiusScalingFactor;

        p.min_radius = min(oldRadiusBound, newRadiusBound);
        p.max_radius = max(oldRadiusBound, newRadiusBound);
        if(oldRadiusBound < newRadiusBound)
            p.growthRate = 1;
        else
            p.growthRate = -1;

        p.timeSinceLastChanged = 0;
        p.lastSoundEvent = currentSound;
    }

    double newRadius;
    if (p.growthRate > 0 && c.radius < p.max_radius) { 
        c.radius = c.radius +  (p.max_radius - c.radius) / 2;
        //newRadius =  p.min_radius + atanf((double) p.timeSinceLastChanged / (double) 10) * (p.max_radius - p.min_radius) * 2/3.14;
        //c.radius = (int) newRadius;
    } else if (p.growthRate < 0 && c.radius > p.min_radius) {
         c.radius = c.radius -  (c.radius - p.max_radius) / 2;

        //newRadius =  p.max_radius - atanf((double) p.timeSinceLastChanged / (double) 10) * (p.max_radius - p.min_radius) * 2/3.14;
        //c.radius = (int) newRadius;
    }
    
    //printf("%d \n", newRadius);

    g.shape.circle = c;
    g.properties = p;

    //printf("displaying thing \n");
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
    P.gBlocks.z = 1;          // only 2D of blocks allowe

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


GPU_Geometries  initGPUGeometries(int size, Geometry *cpuGeometries)
{
    GPU_Geometries G;
    G.size = size;

    int mem_size;
    mem_size = size * sizeof(Geometry);
    cudaMalloc((void **) &G.geometries, mem_size);

    cudaMemcpy(G.geometries, cpuGeometries, mem_size, cH2D);


    //cudaChannelFormatDesc desc = cudaCreateChannelDesc<Geometry>();

    //cudaBindTexture(NULL, texGeometries, G.geometries, mem_size);

    return G;
}

GPU_Midi initGPUMidi(DiscreteTracks dts)
{
    GPU_Midi m;
    m.numTracks = dts.numTracks;
    m.trackLength = dts.trackLength;

    int row_mem_size;
    row_mem_size = dts.trackLength * sizeof(SoundEvent);

    cudaMalloc((void**) &m.tracks, dts.numTracks * row_mem_size);   

    // flatten array and copy into CUDA
    for (int i = 0; i < dts.numTracks; i++) {
        int flatIndex = i * dts.trackLength;
        cudaMemcpy(&(m.tracks[flatIndex]), dts.tracks[i], row_mem_size, cH2D);
    } 
    


    return m;


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
