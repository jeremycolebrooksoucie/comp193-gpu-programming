#ifndef hInterfaceLib
#define hInterfaceLib

#include "params.h"
#include "gpu_main.h"
#include "geometry.h"
int openBMP(AParams *PARAMS, GPU_Palette* P1, char* fileName);
int initGeometries(AParams* p, GPU_Geometries* G1);

int runIt(GPU_Palette* P1, GPU_Geometries *G1, AParams* PARAMS);

int setDefaults(AParams *PARAMS);
int usage();
int viewParams(const AParams *PARAMS);
char crack(int argc, char** argv, char* flags, int ignore_unknowns);
Point newPoint(int, int);
Color newColor(float, float, float); //r, g, b

#endif