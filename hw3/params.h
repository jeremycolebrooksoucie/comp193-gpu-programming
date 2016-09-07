#ifndef hParamsLib
#define hParamsLib


#define THREADS_PER_BLOCK 32




typedef struct AParams {        
  
  int   height;
  int   width;
  int   size;

  int print_height;
  int print_width;
  int print_size;

  char *midiFile;
  char *bmpFile;
  int samplingRate;
  int duration;
  int verbosity;
  int afterimage;

  int curFrame;
} AParams;


#endif