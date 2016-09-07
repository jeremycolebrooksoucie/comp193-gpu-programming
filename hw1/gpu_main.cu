/**************************************************************************
*
*           COMP 193
*           GPU programming 
*           Exercise 1 template 
*
**************************************************************************/

#include <cuda.h>
#include <curand.h>                 // includes random num stuff
#include <curand_kernel.h>       	// more rand stuff
#include <cuda_texture_types.h>
#include "book.h"

#include <stdio.h>
#include "gpu_main.h"

/*************************************************************************/
void addGPU(int *a, int *b, int *c, unsigned long vecSize){

    printf("you can remove this print statement\n");
    // arrays to pass to gpu
    int *dev_a, *dev_b, *dev_c;
    HANDLE_ERROR( cudaMalloc( (void **) &dev_a, vecSize * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void **) &dev_b, vecSize * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void **) &dev_c, vecSize * sizeof(int) ) );

    // copy into these arrays
    HANDLE_ERROR( cudaMemcpy( dev_a, a, vecSize * sizeof(int), 
                               cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMemcpy( dev_b, b, vecSize * sizeof(int), 
                                cudaMemcpyHostToDevice));

    add<<<vecSize, 1>>>( dev_a, dev_b, dev_c, vecSize);

    // copy back to device to fill c with results
    HANDLE_ERROR( cudaMemcpy( c, dev_c, vecSize * sizeof(int), 
                              cudaMemcpyDeviceToHost) );
    // memory clean up
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaFree( dev_b ) );
    HANDLE_ERROR( cudaFree( dev_c ) );

  
}

/*************************************************************************/

/*
 * kernal function to add arrays in a and b
 */
__global__ void add(int *a, int *b, int *c, unsigned long vecSize)
{
    int tid = blockIdx.x;
    if (tid < vecSize) {
        c[tid] = a[tid] + b[tid];
    }
}
