#ifndef hInterfaceLib
#define hInterfaceLib

#include "params.h"
#include "gpu_main.h"
#include "geometry.h"
int openBMP(AParams *PARAMS, GPU_Palette* P1);
int openMidi(AParams *p, GPU_Midi *m);
int initGeometries(GPU_Geometries *gpuPallete, const AParams params, const GPU_Midi gpuMidi);
int runIt(GPU_Palette gpuPallete, GPU_Geometries gpuGeometries, GPU_Midi gpuMidi, AParams params);

AParams getDefaultParameters();
void viewParams(const AParams params);

Point newPoint(int, int);
Color newColor(float, float, float); //r, g, b
Geometry randomGeometry(AParams params);

#endif