/**************************************************************************
*
*           COMP 193
*           GPU programming 
*           Exercise 2 template 
*
**************************************************************************/

#include <cuda.h>
//#include <curand.h>                 // includes random num stuff
//#include <curand_kernel.h>       	// more rand stuff
//#include <cuda_texture_types.h>

#include <stdio.h>
#include "gpu_main.h"
#include "params.h"

// define texture memory
//texture<float, 2> texGray;
texture<float, 2> texBlue;
//texture<float, 2> texGreen;
//texture<float, 2> texBlue;

/*************************************************************************/
int updatePalette(GPU_Palette* P, const float* inPtr){

  float GAIN = 10.0;
    
  float val = inPtr[0];
  val *= GAIN; // put some gain on the val
  if (val > 1.0) val = 1.0; // clip if val is greater than 1
  
//  printf("val = %f\n", val);
  
  updateReds <<< P->gBlocks, P->gThreads >>> (P->red, P->rand, val);
  updateGreens <<< P->gBlocks, P->gThreads >>> (P->green, P->rand, val);
  updateBlues <<< P->gBlocks, P->gThreads >>> (P->blue, P->rand, val);

  return 0;
}
  

/*************************************************************************/
__global__ void updateReds(float* red, curandState* gRand, float amp){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);


  // generate noise & update seed
  curandState localState = gRand[vecIdx];
  float theRand = curand_uniform(&localState);
  gRand[vecIdx] = localState; 
  
  // add change in signal and noise to the signal
  red[vecIdx] = theRand * amp ;
}

/*************************************************************************/
__global__ void updateGreens(float* green, curandState* gRand, float amp){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);


  // generate noise & update seed
  curandState localState = gRand[vecIdx];
  float theRand = curand_uniform(&localState);
  gRand[vecIdx] = localState; 
  
  // add change in signal and noise to the signal
  green[vecIdx] = theRand * amp ;
}

/*************************************************************************/
__global__ void updateBlues(float* blue, curandState* gRand, float amp){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);


  // generate noise & update seed
  curandState localState = gRand[vecIdx];
  float theRand = curand_uniform(&localState);
  gRand[vecIdx] = localState; 
  
  // add change in signal and noise to the signal
 // blue[vecIdx] = theRand * amp * blue[vecIdx];
  blue[vecIdx] = theRand * amp;
}



/*************************************************************************/
// use this for initializing random num generator
__global__ void setup_kernel(curandState* state, unsigned long seed) {

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  curand_init(seed, vecIdx, 0, &state[vecIdx]);
}


/*************************************************************************/
GPU_Palette initGPUPalette(AParams* PARAMS){

  // load
  GPU_Palette P;
  
  P.gTPB = 32;      // threads per block
  P.gDIM = 800;     // assumes the image is 800x800
          
  // 800x800 palette = 25x25 grid of 32x32 threadblocks
  P.gSize = P.gDIM * P.gDIM * sizeof(float);
  
  
  P.gThreads.x = P.gTPB;
  P.gThreads.y = P.gTPB;
  P.gThreads.z = 1;         // 3D of threads allowed
  P.gBlocks.x = P.gDIM/P.gTPB;
  P.gBlocks.y = P.gDIM/P.gTPB;
  P.gBlocks.z = 1;          // only 2D of blocks allowed

  // allocate memory for rand seeds
  unsigned long randSize = P.gDIM * P.gDIM * sizeof(curandState);
	cudaMalloc((void**) &P.rand, randSize);
  
  // allocate memory for the palette
  cudaMalloc((void**) &P.gray, P.gSize);    // black and white (avg of rgb)
  cudaMalloc((void**) &P.red, P.gSize);   // r
  cudaMalloc((void**) &P.green, P.gSize); // g
  cudaMalloc((void**) &P.blue, P.gSize);  // b
    
  // create texture memory and bind to black and white data
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
//  cudaBindTexture2D(NULL, texGray, P.gray, desc, P.gDIM,
//                        P.gDIM, sizeof(float) * P.gDIM);

  cudaBindTexture2D(NULL, texBlue, P.blue, desc, P.gDIM,
                        P.gDIM, sizeof(float) * P.gDIM);
  
  // init rands on gpu
  setup_kernel <<< P.gBlocks, P.gThreads >>> (P.rand, time(NULL));
  
  return P;
}





/*************************************************************************/
int freeGPUPalette(GPU_Palette* P) {

  // free texture memory
//  cudaUnbindTexture(texGray); // this is bound to black and white
  cudaUnbindTexture(texBlue); // this is bound to black and white
  
  // free gpu memory
  cudaFree(P->gray);
  cudaFree(P->red);  
  cudaFree(P->green);
  cudaFree(P->blue);  
  
  cudaFree(P->rand);
  
  return 0;
}

/*************************************************************************/
