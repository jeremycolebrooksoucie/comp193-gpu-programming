/**************************************************************************
*
*           COMP 193
*           GPU programming 
*           Exercise 1 template 
*
**************************************************************************/
#include <stdio.h>
#include <cstdlib>
#include <string.h>
#include <time.h>

#include "interface.h"
#include "gpu_main.h"

// these needed to 'crack' the command line
int		arg_index = 0;	//	
char	*arg_option;
char	*pvcon = NULL;


/*************************************************************************/
int main(int argc, char *argv[]){

	unsigned char ch;
  clock_t start, end;
	AParams PARAMS;
  
  setDefaults(&PARAMS);
  
  // -- get parameters that differ from defaults from command line:
	while((ch = crack(argc, argv, "r|v|s|", 0)) != NULL) {
	  switch(ch){
    	case 'r' : PARAMS.runMode = atoi(arg_option); break; 
        case 'v' : PARAMS.verbosity = atoi(arg_option); break;
        case 's' : PARAMS.vecSize = atoi(arg_option); break;
        default  : usage(); return(0);
    }
  }

  if (PARAMS.verbosity == 2) viewParams(&PARAMS);
  
  // allocate memory on CPU for three arrays
  int a[PARAMS.vecSize], b[PARAMS.vecSize], c[PARAMS.vecSize];
     
  // fill the cpu arrays with numbers
  for (int i = 0; i < PARAMS.vecSize; i++) {
      a[i] = -i;
      b[i] = i * i;
  }
  
  // -- run the system depending on runMode
  switch(PARAMS.runMode){
      case 1: // add the numbers on the CPU
          if (PARAMS.verbosity)
              printf("\n -- running on CPU -- \n");
          start = clock();
          addCPU(a, b, c, &PARAMS);
          end = clock();
          break;
          
      case 2: // run GPU version
          if (PARAMS.verbosity)
              printf("\n -- running on GPU -- \n");
          start = clock();
          addGPU(a, b, c, PARAMS.vecSize);
          end = clock();
          break;

      default: printf("no valid run mode selected\n");
  }

  // print the result if in verbose mode
  if (PARAMS.verbosity)
    for (int i=0; i<PARAMS.vecSize; i++) {
      printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }

  // print the time used
  printf("time used: %.2f\n", ((double)(end - start))/ CLOCKS_PER_SEC);
 
return 0;
}



/**************************************************************************
                       PROGRAM FUNCTIONS
**************************************************************************/
void addCPU(int *a, int *b, int *c, AParams *PARAMS){
    int tid = 0;    // this is CPU zero, so we start at zero
    while (tid < PARAMS->vecSize) {
        c[tid] = a[tid] + b[tid];
        tid += 1;   // we have one CPU, so we increment by one
    }
}



/**************************************************************************
                       INTERFACE HELPER FUNCTIONS
**************************************************************************/
int setDefaults(AParams *PARAMS){

    PARAMS->verbosity   = 1;
    PARAMS->runMode     = 1;   
    PARAMS->vecSize     = 100000; // hundred thousand

    return 0;     
}

/*************************************************************************/
int usage()
{
	printf("USAGE:\n");
	printf("-f[val] -r[val] -v[val] -s[val] \n\n"); 
  printf("e.g.> ex1 -r1 -v1 -s1000\n");
  printf("v  verbose mode (0:none, 1:normal, 2:params\n");
  printf("r  run mode (1:CPU, 2:GPU)\n"); 
  printf("s  vector size\n");
  
  return(0);
}

/*************************************************************************/
int viewParams(const AParams *PARAMS){
    
    printf("--- PARAMETERS: ---\n");

    // verbose mode = 2 to have gotten here
    
    printf("run mode: %d\n", PARAMS->runMode); 
    printf("vec size: %lu\n", PARAMS->vecSize);
    
    return 0;
}

/*************************************************************************/
/* 
    Globals
    int     arg_index = 0;  // index of current flag
    char    *arg_option;    // current input flag code is examining
    char    *pvcon = NULL;  // pointer to current flag to parse
 */
char crack(int argc, char** argv, char* flags, int ignore_unknowns)
{
    char *pv, *flgp;

    // examing all valid arguments
    while ((arg_index) < argc) {
        // pick up where you left off
        if (pvcon != NULL) {
            pv = pvcon;
        }
        else {
            // don't go outside bounds
            if (++arg_index >= argc) {
                return(NULL); 
            }
            // set pv to current arg string
            pv = argv[arg_index];
            // ignore the flag if its not valid (doesn't begin with -)
            if (*pv != '-') {
                return(NULL);
            }
        }
        // look at characters after the flag. 
        pv++;

        if (*pv != NULL) {
            // search for the first character in flags
            if ((flgp=strchr(flags,*pv)) != NULL) {
                pvcon = pv;                     
                if (*(flgp+1) == '|') {
                    // arg_options should contain string with int value
                    arg_option = pv+1; 
                    pvcon = NULL; 
                }
                return(*pv);
            }
            else if (!ignore_unknowns){
                    fprintf(stderr, "%s: no such flag: %s\n", argv[0], pv);
                    return(EOF);
            }
            else {
                pvcon = NULL;
            }
	    }
            pvcon = NULL;
    }
    return(NULL);
}

/*************************************************************************/


