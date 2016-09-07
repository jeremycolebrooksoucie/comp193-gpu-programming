/**************************************************************************
 *
 *	The MyAiff Class
 *	Copyright 2002 - Michael Brady, Fluidbase Systems
 *
 **************************************************************************/
#ifndef hMyAiff
#define hMyAiff

#include <stdio.h>
//#include "bradylib.h"

//#define TWO_PI              6.28319
#define TRUE                1
#define FALSE               0
#define MYBUFSIZ            4096


/*************************************************************************/
class AMyAiff{
public:
  // functions
  AMyAiff();  // constructor
  unsigned short 	ScanHeader(FILE*);
  unsigned short 	WriteHeader(FILE*);
  unsigned short  WriteLengthInfo(FILE*, unsigned long);
  
  // params
  short           fNumChannels;
  unsigned long   fNumSampleFrames;
  short           fSampleSize;
  unsigned long   fSampleRate;
  unsigned long   fOffset;
  unsigned long   fBlocksize;
  
private:
  // functions
  unsigned short 	GetHeaderInfo(FILE*);
  unsigned short 	ExamineNextChunk(FILE*);
  unsigned short 	ReadCommInfo(FILE*);
  unsigned short 	ReadSsndInfo(FILE*);
  unsigned short 	CheckForInputErrors();
  
  // params
  unsigned short fHaveComInfo;
  long   fMarker1;
  long   fMarker2;
  long   fMarker3;
};
/******************************************************************************/
#endif