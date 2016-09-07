#ifndef GPULib
#define GPULib

#include <cuda.h>
#include <curand.h>                 // includes random num stuff
#include <curand_kernel.h>       	// more rand stuff
#include <cuda_texture_types.h>

// regular CPU function
void addGPU(int *a, int *b, int *c, unsigned long vecSize);

// GPU kernel function
__global__ void add(int *a, int *b, int *c, unsigned long vecSize);


#endif  // GPULib

