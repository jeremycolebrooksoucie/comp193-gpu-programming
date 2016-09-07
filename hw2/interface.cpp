#include <stdio.h>
#include <cstdlib>
#include <string.h>
#include <time.h>

#include "interface.h"
#include "gpu_main.h"
#include "params.h"
#include "animate.h"
#include "qdbmp.h"

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

  // read in the image file, write the data to the gpu:
	int err = openBMP(&PARAMS, &P1, fileName);
  
  // view parameters if specified by verbose level:
  if (PARAMS.verbosity == 2) viewParams(&PARAMS);
    
  // -- run the system depending on runMode
  switch(PARAMS.runMode){
      case 1: // add the numbers on the CPU
          if (PARAMS.verbosity)
              printf("\n -- doing something in runmode 1 -- \n");
          start = clock();
          runIt(&P1, &PARAMS); // always run on GPU because CPU note supported
          end = clock();
          break;
          
      case 2: // run GPU version
          if (PARAMS.verbosity)
              printf("\n -- doing somethign in runmode 2 -- \n");
          start = clock();
          runIt(&P1, &PARAMS);
          end = clock();
          break;

      default: printf("no valid run mode selected\n");
  }

  // print the time used
  printf("time used: %.2f\n", ((double)(end - start))/ CLOCKS_PER_SEC);
 
return 0;
}


/**************************************************************************
                       PROGRAM FUNCTIONS
**************************************************************************/
int runIt(GPU_Palette* P1, AParams* PARAMS){
 
  printf("runit \n");
	CPUAnimBitmap animation(PARAMS, P1);
  printf("animiation instantion successful\n");

  animation.initAnimation();
  reduceToEdges(P1);
  colorAroundEdges(P1);
  int i = 0;
  while(1){
    if (PARAMS->verbosity)
      printf("Iteration: %d \n", i++);
    spreadColor(P1);
    animation.drawPalette();
  }
  
  return(0);
}

/**************************************************************************
                       PROGRAM FUNCTIONS
**************************************************************************/
// this function loads in the initial picture to process
int openBMP(AParams* PARAMS, GPU_Palette* P1, char* fileName){

BMP* bmp;
bmp = BMP_ReadFile(fileName);
BMP_CHECK_ERROR( stderr, -1 );

PARAMS->print_width = BMP_GetWidth( bmp );
PARAMS->print_height = BMP_GetHeight( bmp );
// use integer mult and div to round to nearest multiple of THREADS_PER_BLOCK
PARAMS->width =  ((PARAMS->print_width  + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK) * THREADS_PER_BLOCK;
PARAMS->height = ((PARAMS->print_height + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK) * THREADS_PER_BLOCK;

PARAMS->size = 3 * PARAMS->width * PARAMS->height;
PARAMS->print_size = 3 * PARAMS->print_width * PARAMS->print_height;


(*P1) = initGPUPalette(PARAMS);

// need to use calloc to have 'extra' cells be empty, this requires
// some manipulations of parameters
float* graymap = (float*) calloc(P1->gSize / sizeof(float), sizeof(float));
float* redmap = (float*) calloc(P1->gSize / sizeof(float), sizeof(float));
float* greenmap = (float*) calloc(P1->gSize / sizeof(float), sizeof(float));
float* bluemap = (float*) calloc(P1->gSize / sizeof(float), sizeof(float));



for (int x = 0; x < PARAMS -> print_width; x++)
{
    for (int y = 0; y < PARAMS -> print_height; y++)
    {
        unsigned char r, g, b;
        int tmp_y = PARAMS -> print_height - (y + 1);//needed to not invert image
        BMP_GetPixelRGB(bmp, x, tmp_y, &r, &g, &b);

        // segregate data as floats between 0 - 1 to be written to gpu
        int graymapIdx = x + y * PARAMS -> width;
        graymap[graymapIdx]   = (float) (r + g + b)/(255.0*3.0);
        redmap[graymapIdx]    = (float) r/255.0;
        greenmap[graymapIdx]  = (float) g/255.0;
        bluemap[graymapIdx]   = (float) b/255.0;
    }
}
BMP_Free( bmp );


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




/**************************************************************************
                       INTERFACE HELPER FUNCTIONS
**************************************************************************/
int setDefaults(AParams *PARAMS){

    PARAMS->verbosity       = 1;
    PARAMS->runMode         = 1;
    
    PARAMS->height     = 800;
    PARAMS->width      = 800;
    PARAMS->size      = 800*800*3; // 800x800 pixels x 3 colors
    
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
int viewParams(const AParams *PARAMS){
    
    printf("--- PARAMETERS: ---\n");
    
    printf("run mode: %d\n", PARAMS->runMode);
    
    printf("image height: %d\n", PARAMS->height);
    printf("image width: %d\n", PARAMS->width);
    printf("data size: %d\n", PARAMS->size);
    
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


