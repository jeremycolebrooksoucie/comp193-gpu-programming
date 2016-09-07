#ifndef hParamsLib
#define hParamsLib


#define THREADS_PER_BLOCK 32




typedef struct AParams {        
  int   verbosity;
  int   runMode;
  
  int   height;
  int   width;
  int   size;

  int print_height;
  int print_width;
  int print_size;
} AParams;


#endif