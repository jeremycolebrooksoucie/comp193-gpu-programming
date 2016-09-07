/**************************************************************************
*
*           COMP 193
*           GPU programming 
*           Exercise 3 template 
*
**************************************************************************/
#include <stdio.h>
//#include <stdlib.h>
#include <cstdlib>
#include <string.h>
#include <time.h>
#include <math.h>

#include "interface.h"
#include "gpu_main.h"
#include "params.h"
#include "animate.h"
#include "myAIFF.h"

// has cuda fft stuff
//#include <cuda_runtime.h>
//#include <cufft.h>
//#include <helper_functions.h>
//#include <helper_cuda.h>

// define fftData to hold complex numbers
//typedef float2 fftData;


// these needed to 'crack' the command line
int		arg_index = 0;		
char	*arg_option;
char	*pvcon = NULL;



/*************************************************************************/
int main(int argc, char *argv[]){

	unsigned char ch;
  clock_t start, end;
	AParams PARAMS;
  
  setDefaults(&PARAMS);
    
  // -- get parameters that differ from defaults from command line:
  if(argc<2){usage(); return 1;} // must be at least one arg (fileName)
	while((ch = crack(argc, argv, "r|v|", 0)) != NULL) {
	  switch(ch){
    	case 'r' : PARAMS.runMode = atoi(arg_option); break; 
      case 'v' : PARAMS.verbosity = atoi(arg_option); break;
      default  : usage(); return(0);
    }
  }

  // filename of image.bmp should be last arg on command line:
  char* fileName = argv[arg_index];
  
  // initialize memory on the gpu
  GPU_Palette P1;
  P1 = initGPUPalette(&PARAMS);
  
  // read in the image file, write the data to the gpu:
//  char* imageName = "SN.bmp";
//	int err = openBMP(&PARAMS, &P1, imageName);
  
  int err = initPalette(&PARAMS, &P1);
  
  // read in the sound file
	FILE *infp;
  if((infp = fopen(fileName,"r+b")) == NULL){
      printf("Can't open input file: %s\n", fileName);
      return(1);
      }
  PARAMS.num_ms_per_frame = 10; // duration of a tick
  
  // view parameters if specified by verbose level:
//  if (PARAMS.verbosity == 2) viewParams(&PARAMS);
    
  // -- run the system depending on runMode
  switch(PARAMS.runMode){
      case 1: // do it one way
          if (PARAMS.verbosity)
              printf("\n -- doing something in runmode 1 -- \n");
          runIt(infp, &P1, &PARAMS);
          break;
          
      case 2: // do it some other way
          if (PARAMS.verbosity)
              printf("\n -- doing something in runmode 2 -- \n");
          break;

      default: printf("no valid run mode selected\n");
  }

  // print the time used
//  printf("time used: %.2f\n", ((double)(end - start))/ CLOCKS_PER_SEC);
 
return 0;
}


/**************************************************************************
                       PROGRAM FUNCTIONS
**************************************************************************/
int runIt(FILE* infp, GPU_Palette* P1, AParams* PARAMS){
 
  // suet up palette for animation
	CPUAnimBitmap animation(P1->gDIM, P1->gDIM, P1);
  cudaMalloc((void**) &animation.dev_bitmap, animation.image_size());
  animation.initAnimation();

  // info for reading in an aiff file
  AMyAiff theAIFF;
  
  // scan up until the beginning of the sound data
  int err = 0;
  err = theAIFF.ScanHeader(infp);
  if (err) return (err);
  
  // anticipate number of time steps to run, assuming 10ms tick
  err = getAIFFinfo(&theAIFF, PARAMS);
	if (err) return (err); 
  
  int SigSize = PARAMS->fSizeOfFrame/4; // should be 441 if 10ms tick
  
	if (PARAMS->verbosity == 2) viewParams(PARAMS, &theAIFF);
  
  signed long NumBytesRead;
  char inputBuffer[MYBUFSIZ];
  
  // pretend we have 12 frequency bands:
  int NumFreqBands = 12;
	float theFreqBands[NumFreqBands];
  
  for (unsigned long frame = 0; frame < PARAMS->fNumFrames; frame++){
    
    // --- read in the sound sample
    NumBytesRead = fread(inputBuffer, 1, PARAMS->fSizeOfFrame, infp);
    if(NumBytesRead != PARAMS->fSizeOfFrame){
      printf("reading last partial frame, but shouldn't get here\n");
      return(1);
    }
      
    // get the crude height of the amplitude envelope for the sample:
    float acc = 0;
    float val;
    for (int i = 0; i < SigSize; i++){
      // read msb (most significant byte) of ch 1, ignore other 3 bytes
      val =  (float)(inputBuffer[i*4])/127.0;
      if (val < 0) val *= -1; // abs(x) doesn't work for some reason?!
//      printf("%f\n", val);
      acc += val;
      }
    acc /= SigSize;
    
    // fill freq bands with same value
    for (int fb = 0; fb < NumFreqBands; fb++){
      theFreqBands[fb] = acc;
    }
        
    // update the palette based on the amp envelope val (acc)
    int err = updatePalette(P1, theFreqBands);
    
    animation.drawPalette(P1->gDIM, P1->gTPB);
  }
  
  return(0);
}

/**************************************************************************
                       PROGRAM FUNCTIONS
**************************************************************************
int runIt(FILE* infp, GPU_Palette* P1, AParams* PARAMS){
 
	CPUAnimBitmap animation(P1->gDIM, P1->gDIM, P1);
  cudaMalloc((void**) &animation.dev_bitmap, animation.image_size());
  animation.initAnimation();

  AMyAiff theAIFF;
  
  // scan up until the beginning of the sound data
  int err = 0;
  err = theAIFF.ScanHeader(infp);
  if (err) return (err);
  
  // figure out how many time steps there will be, and how many samples
  // to read per time step:
  err = getAIFFinfo(&theAIFF, PARAMS);
	if (err) return (err); 
  
  int SigSize = PARAMS->fSizeOfFrame/4; // 144
  
  // Allocate host memory for the signal
  fftData* h_signal = (fftData*) malloc(sizeof(fftData) * SigSize);
  
	// Allocate readout memory for the signal
  fftData* h_transformed = (fftData*) malloc(sizeof(fftData) * SigSize);


  // Initialize host memory for the signal
  for (unsigned int i = 0; i < SigSize; ++i)
  {
    h_signal[i].x = rand() / (float)RAND_MAX;
    h_signal[i].y = 0;
  }
  
  // Allocate device memory for the signal
  fftData* d_signal;
  cudaMalloc((void **) &d_signal, sizeof(fftData) * SigSize);
  
  // copy host memory to device
  cudaMemcpy(d_signal, h_signal, SigSize, cH2D);
  
	if (PARAMS->verbosity == 2) viewParams(PARAMS, &theAIFF);
  
  signed long NumBytesRead;
  char inputBuffer[MYBUFSIZ];
  
  for (unsigned long frame = 0; frame < PARAMS->fNumFrames; frame++){
    
    
    // --- this reads in the sound sample..
    // get sound data for the time step:
    NumBytesRead = fread(inputBuffer, 1, PARAMS->fSizeOfFrame, infp);
    if(NumBytesRead != PARAMS->fSizeOfFrame){
      printf("reading last partial frame, but shouldn't get here\n");
      return(1);
    }
      
    // write data to GPU
    for (unsigned int i = 0; i < SigSize; ++i){
      // just need every 4th byte (MSB of ch. 1), divide by 127:
      h_signal[i].x = (float) abs(inputBuffer[i*4]/127.0); 
      h_signal[i].y = 0;
      }

    cudaMemcpy(d_signal, h_signal, SigSize, cH2D);
    
    // CUFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, SigSize, CUFFT_C2C, 1);

    // transform signal
    cufftExecC2C(plan, (cufftComplex *)d_signal, 
                              (cufftComplex *)d_signal, CUFFT_FORWARD);
    
    cudaMemcpy(h_transformed, d_signal, SigSize, cD2H);
    
    // print to screen
    printf("next FFT:\n"); 
    for (unsigned int i = 0; i < SigSize; ++i){
      // just need every 4th byte (MSB of ch. 1):
      printf("%f\n", h_transformed[i].x); 
//      printf("%d\n", inputBuffer[i*4]);       
    }

    // --- that all did prepped the sound sample as an FFT
    
    int err = updatePalette(P1);
  
    animation.drawPalette(P1->gDIM, P1->gTPB);
  }
  
  return(0);
}

/*************************************************************************/
int getAIFFinfo(AMyAiff* inAiff, AParams* PARAMS)
{
  PARAMS->fSizeOfFrame =  // 441 * 4 = 1764
   (long)(inAiff->fNumChannels * (inAiff->fSampleSize / 8.0) * 
        inAiff->fSampleRate * PARAMS->num_ms_per_frame) / 1000.0;

  PARAMS->fNumFrames = 
          (inAiff->fNumSampleFrames * inAiff->fNumChannels * 
              (inAiff->fSampleSize / 8)) / PARAMS->fSizeOfFrame;
   
  PARAMS->fNumBytesPerSamp = 
          inAiff->fNumChannels * (inAiff->fSampleSize / 8);
  
  PARAMS->fBuffSize = PARAMS->fSizeOfFrame/PARAMS->fNumBytesPerSamp;

  unsigned long fFftSize;
  fFftSize = 2;
  while(fFftSize < PARAMS->fBuffSize) fFftSize *= 2;

  PARAMS->fOverlapSize = 
          (fFftSize * PARAMS->fNumBytesPerSamp) - PARAMS->fSizeOfFrame;

  printf("FFTSize: %d\n", fFftSize);

  if(PARAMS->fSizeOfFrame > MYBUFSIZ){
    printf("need to allocate a larger buffer size (MYBUFSIZ):\n");
    printf("Size of input buffer = %d bytes\n", PARAMS->fSizeOfFrame);
    printf("Size of memory buffer = %d bytes\n", MYBUFSIZ);
    return(1);
    }
  
  return(0);    
}




/*************************************************************************
// this function loads in the initial picture to process
int openBMP(AParams* PARAMS, GPU_Palette* P1, char* fileName){

// open the file
FILE *infp;
if((infp = fopen(fileName, "r+b")) == NULL){
	printf("can't open image of filename: %s\n", fileName);
  return 0;
}

// read in the 54-byte header
unsigned char header[54];
fread(header, sizeof(unsigned char), 54, infp);
PARAMS->width = *(int*)&header[18];
PARAMS->height = *(int*)&header[22];
PARAMS->size = 3 * PARAMS->width * PARAMS->height;

// read in the data
unsigned char data[PARAMS->size];
fread(data, sizeof(unsigned char), PARAMS->size, infp);
fclose(infp);

float* graymap = (float*) malloc(P1->gSize);
float* redmap = (float*) malloc(P1->gSize);
float* greenmap = (float*) malloc(P1->gSize);
float* bluemap = (float*) malloc(P1->gSize);

for(int i = 0; i < PARAMS->size; i += 3)
{
  // flip bgr to rgb (red - green - blue)
  unsigned char temp = data[i];
  data[i] = data[i+2];
  data[i+2] = temp;
  
  // segregate data as floats between 0 - 1 to be written to gpu
  int graymapIdx = (int) floor(i/3.0);
  graymap[graymapIdx]   = (float) (data[i]+data[i+1]+data[i+2])/(255.0*3.0);
  redmap[graymapIdx]    = (float) data[i]/255.0;
  greenmap[graymapIdx]  = (float) data[i+1]/255.0;
  bluemap[graymapIdx]   = (float) data[i+2]/255.0;
}

// write image data to the GPU
cudaMemcpy(P1->gray, graymap, P1->gSize, cH2D);
cudaMemcpy(P1->red, redmap, P1->gSize, cH2D);
cudaMemcpy(P1->green, greenmap, P1->gSize, cH2D);
cudaMemcpy(P1->blue, bluemap, P1->gSize, cH2D);

// free CPU memory
free(graymap);
free(redmap);
free(greenmap);
free(bluemap);

return 0;
}


/*************************************************************************/
// init palette to random colors
int initPalette(AParams* PARAMS, GPU_Palette* P1){

  
float* graymap = (float*) malloc(P1->gSize);
float* redmap = (float*) malloc(P1->gSize);
float* greenmap = (float*) malloc(P1->gSize);
float* bluemap = (float*) malloc(P1->gSize);


for(int i = 0; i < P1->gSize/sizeof(float); i++)
{
  graymap[i]   = rand() / (float)RAND_MAX;
  redmap[i]    = rand() / (float)RAND_MAX;
  greenmap[i]  = rand() / (float)RAND_MAX;
  bluemap[i]   = rand() / (float)RAND_MAX;
}

// write image data to the GPU
cudaMemcpy(P1->gray, graymap, P1->gSize, cH2D);
cudaMemcpy(P1->red, redmap, P1->gSize, cH2D);
cudaMemcpy(P1->green, greenmap, P1->gSize, cH2D);
cudaMemcpy(P1->blue, bluemap, P1->gSize, cH2D);

// free CPU memory
free(graymap);
free(redmap);
free(greenmap);
free(bluemap);


//float val = 1.0;
//int goo;

//printf("%f as hex %d\n", val, goo);
//cudaMemset(P1->red, 0x55, P1->gSize);

return 0;
}


/**************************************************************************
                       INTERFACE HELPER FUNCTIONS
**************************************************************************/
int setDefaults(AParams *PARAMS){

    PARAMS->verbosity       = 1;
    PARAMS->runMode         = 1;
    
    PARAMS->height     = 800;
    PARAMS->width      = 800;
//    PARAMS->size      = 800*800*3; // 800x800 pixels x 3 colors
    
    return 0;     
}

/*************************************************************************/
int usage()
{
	printf("USAGE:\n");
	printf("-r[val] -v[val] filename\n\n"); 
  printf("e.g.> ex2 -r1 -v1 imagename.bmp\n");
  printf("v  verbose mode (0:none, 1:normal, 2:params\n");
  printf("r  run mode (1:CPU, 2:GPU)\n"); 
  
  return(0);
}

/*************************************************************************/
int viewParams(const AParams *PARAMS, AMyAiff* inAiff){
    
  printf("--- PARAMETERS: ---\n");

  printf("run mode: %d\n", PARAMS->runMode);

  // palette info:
  printf("image height: %d\n", PARAMS->height);
  printf("image width: %d\n", PARAMS->width);
  printf("data size: %d\n", PARAMS->size);

  // sound file info:
  printf("numChannels: %d\n", inAiff->fNumChannels);
  printf("sampleSize: %d\n", inAiff->fSampleSize);
  printf("sampleRate: %d\n", inAiff->fSampleRate);
  printf("num ms per frame: %d\n", PARAMS->num_ms_per_frame);
  printf("size of frame: %d\n", PARAMS->fSizeOfFrame);
	printf("numFrames: %d\n", PARAMS->fNumFrames);
  printf("NumBytesPerSamp: %d\n", PARAMS->fNumBytesPerSamp);  
  printf("BuffSize: %d\n", PARAMS->fBuffSize);
//  printf("FFTSize: %d\n", fFftSize);
  printf("OverlapSize: %d\n", PARAMS->fOverlapSize);

    
    return 0;
}

/*************************************************************************/
char crack(int argc, char** argv, char* flags, int ignore_unknowns)
{
    char *pv, *flgp;

    while ((arg_index) < argc){
        if (pvcon != NULL)
            pv = pvcon;
        else{
            if (++arg_index >= argc) return(NULL); 
            pv = argv[arg_index];
            if (*pv != '-') 
                return(NULL);
            }
        pv++;

        if (*pv != NULL){
            if ((flgp=strchr(flags,*pv)) != NULL){
                pvcon = pv;                     
                if (*(flgp+1) == '|') { arg_option = pv+1; pvcon = NULL; }
                return(*pv);
                }
            else
                if (!ignore_unknowns){
                    fprintf(stderr, "%s: no such flag: %s\n", argv[0], pv);
                    return(EOF);
                    }
                else pvcon = NULL;
	    	}
            pvcon = NULL;
            }
    return(NULL);
}

/*************************************************************************/


