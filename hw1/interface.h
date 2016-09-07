#ifndef hInterfaceLib
#define hInterfaceLib

class AParams {
public:
        
  int             verbosity;
  int             runMode;
  unsigned long   vecSize;

};
  
void addCPU(int *a, int *b, int *c, AParams *PARAMS);

int setDefaults(AParams *PARAMS);
int usage();
int viewParams(const AParams *PARAMS);
char crack(int argc, char** argv, char* flags, int ignore_unknowns);


#endif