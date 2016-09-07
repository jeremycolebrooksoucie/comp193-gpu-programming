#ifndef hInterfaceLib
#define hInterfaceLib

#include "params.h"
#include "gpu_main.h"
#include "myAIFF.h"


//int openBMP(AParams *PARAMS, GPU_Palette* P1, char* fileName);
int initPalette(AParams *PARAMS, GPU_Palette* P1);
float randFrac(void);

int getAIFFinfo(AMyAiff* theAIFF, AParams* PARAMS);
  
int runIt(FILE*, GPU_Palette* P1, AParams* PARAMS);

int setDefaults(AParams *PARAMS);
int usage();
//int viewParams(const AParams *PARAMS);
int viewParams(const AParams* PARAMS, AMyAiff* inAiff);
char crack(int argc, char** argv, char* flags, int ignore_unknowns);





#endif