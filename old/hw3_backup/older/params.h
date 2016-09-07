#ifndef hParamsLib
#define hParamsLib

class AParams {
public:
        
  int   verbosity;
  int   runMode;
  
  int   height;
  int   width;
  int   size;
  
	int             num_ms_per_frame;
  
  unsigned long   fSizeOfFrame;
  unsigned long   fNumFrames;
  unsigned short  fNumBytesPerSamp;
  unsigned long   fBuffSize;
  unsigned long   fFftSize;
  unsigned long   fOverlapSize;
  
};


#endif