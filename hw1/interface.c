/**************************************************************************
*
*           COMP 193
*           GPU programming 
*           Exercise 1 template 
*
**************************************************************************/
#include <stdio.h>
#include <cstdlib>
#include "interface.h"

// these needed for the command-line interface
int		arg_index = 0;		
char	*arg_option;
char	*pvcon = NULL;


/*************************************************************************/
int main (int argc, char *argv[]){

	unsigned char ch;
	AParams PARAMS;
//  setDefaults(&PARAMS);
  char fileName[80];
  
  int verbosity;
  int runMode;
  
  runMode = 1;
  verbosity = 1;

  // get parameters that differ from defaults
//	if (argc < 2){usage(); return 1;}
	while((ch = crack(argc, argv, "f|r|v|", 0)) != NULL) {
	  switch(ch){
    	case 'f' : sprintf(fileName, "Audio/%s", arg_option); break;
    	case 'r' : runMode = atoi(arg_option); break; 
      case 'v' : verbosity = atoi(arg_option); break;
      default  : usage(); return(0);
    }
  }
    
  // -- seed randoms
//  time_t x;
//  time(&x);
//  srand(x);    

//  PARAMS.crud = 1;
  
  // -- if general verbosity on, view params now that everything is set:
//  if (PARAMS.verbosity == 1) viewParams(&PARAMS);

 
  // ---- run the system depending on runMode
  // RENUMBER THESE TO BETTER ORDER
  switch(runMode){
      case 1: // run CPU version
          if (verbosity == 1)
              printf("\n -- running on CPU -- \n");
          break;
      case 2: // run GPU version
          if (verbosity == 1)
              printf("\n -- running on GPU -- \n");
          break;

      default: printf("no valid run mode selected\n");
  }

//freeField(&F1);

return 0;
}

/*************************************************************************
int runMic(FieldData* F1, const CTCFile* PARAMS){
  int coeffsStart = 0;
  int coeffsEnd = 26;
  if(PARAMS->styleInput == 2){coeffsStart = 0; coeffsEnd = 13;}
  if(PARAMS->styleInput == 3){coeffsStart = 13; coeffsEnd = 26;}
  
  Asr mic(1);
  mic.audio = new Mic("microphoneOutput");
  
  mic.audio->record();
  pthread_create(&mic.audio->writer, NULL, mic.audio->results_helper, 
                                                        (void*) mic.audio);
  mic.listening = true;
  
  while (mic.getListening()) {
    
    if (!mic.audio->recordings.empty()) {
      //pop and save
      
      std::vector<std::vector<double>> theMFCCS=mic.input->transformRaw(
              mic.input->quantizePCM(mic.audio->recordings.front(),false));
      
      for(vector<double> mfccs: theMFCCS){
        vector<float> floatSig(mfccs.begin()+coeffsStart,
                                              mfccs.begin()+coeffsEnd);
        // this is old "old tick"
        DNFsetSigs(F1, 1, floatSig.data());
//     DNFsetSigs(F1, 0, nullSound);
//     DNFrespond(F1);
//     DNFoutput(F1)
        
        for(float coefficent: floatSig){
          std::cout<<coefficent<<" ";
        }
        std::cout<<std::endl;
      }
      
      mic.audio->recordings.pop();
      //cerr<<Audio->recordings.size()<<endl<<endl;
    }
    usleep(1);// i guess we still need this
    
  }
  
  return 0;
}

/*************************************************************************
int openSound(CTCFile* PARAMS, char* soundDirName){
  
  // for deciding which portion ofthe mfccs to use:
  // 1-13 are the root coeffs, 14-26 are the first diffs
  int coeffsStart = 0;
  int coeffsEnd = 26;
  if(PARAMS->styleInput == 2){coeffsStart = 0; coeffsEnd = 13;}
  if(PARAMS->styleInput == 3){coeffsStart = 13; coeffsEnd = 26;}
  
  
  Asr sound(1);
  
  // 1 for loading .wav files, 2 for importing .txt (matlab) files
  if(PARAMS->preProcess == 1) sound.input->loadFiles(soundDirName);
  else sound.input->import(soundDirName);
  
  if(PARAMS->verbosity == 1) sound.data.stats();
  PARAMS->theNames = sound.data.names;
  
  for(int i=0; i<PARAMS->numToruses; i++){
    for(int j=0; j<PARAMS->theMap[i].size(); j++){
      if (std::find(PARAMS->uniquePhones.begin(),
              PARAMS->uniquePhones.end(),
              PARAMS->theMap[i][j]) == PARAMS->uniquePhones.end()){
        PARAMS->uniquePhones.push_back(PARAMS->theMap[i][j]);
      }
    }
  }
  
  // convert input from doubles to floats
  // and only use MFCCs in range specified by PARAMS
  for(int phone=0; phone<sound.data.processed.size(); phone++){
    vector<vector<vector<float>>> tempPhone;
    for(int exemp=0; exemp< sound.data.processed[phone].size(); exemp++){
      vector<vector<float>> tempExemp;
      for(int t=0; t<sound.data.processed[phone][exemp].size();t++){
        vector<float> floatSig(
                sound.data.processed[phone][exemp][t].begin()+coeffsStart,
                sound.data.processed[phone][exemp][t].begin()+coeffsEnd
                );
        tempExemp.push_back(floatSig);
      }
      tempPhone.push_back(tempExemp);
    }
    PARAMS->theData.push_back(tempPhone);
  }
  
  return 0;
}

/*************************************************************************
int isoDNF(FieldData* F1, const CTCFile* PARAMS){
// run basic dynamic neural field with no input - mainly for bug chasing
  
  CPUAnimBitmap animation(F1->gDIM, F1->gDIM, F1);
  if (PARAMS->viewStyle > 0) { // if viewStyle is non-zero, animating:
    cudaMalloc((void**) &animation.dev_bitmap, animation.image_size());
    animation.initAnimation();
  }
  
  // define a modulator node to pass to DNFtick - doesn't get used
  float modNodesPtr[1];
  
  // define input sound for num driver nodes - also doesn't get used
  float nullSound[F1->gNDN];
  
  while(1){ // just run until user kills or closes animation window
    // initialize the driver and modulator signals to noise:
    DNFsetSigs(F1, 0, nullSound);
    DNFrespond(F1);
    
    if (PARAMS->viewStyle > 0)
      animation.AnimateField(PARAMS->viewStyle, F1->gDIM, F1->gTPB);
  }
  
  return 0;
}

/*************************************************************************
int trainDNF(FieldData* F1, const CTCFile* PARAMS, char* fileName) {
  
  CPUAnimBitmap animation(F1->gDIM, F1->gDIM, F1);
  if (PARAMS->viewStyle > 0) { // if viewStyle is non-zero, animating:
    cudaMalloc((void**) &animation.dev_bitmap, animation.image_size());
    animation.initAnimation();
  }
  
  // load the brain - only need to do this once
  loadCTC(fileName, F1, PARAMS);
  
  // (save the brain at the end of each training epoch)
  for (int epoch = 0; epoch < PARAMS->nTrainEpochs; epoch++){
    
    for (int cycle = 0; cycle < PARAMS->nItersPerEpoch; cycle++){
      printf("training epoch: %d, cycle: %d\n", epoch+1, cycle+1);
      
      //create random ordering of recordings
      vector<int> phoneOrdering;
      for (int i = 0; i < F1->gNMN; i++){
        phoneOrdering.push_back(i);
      }
      random_shuffle(phoneOrdering.begin(), phoneOrdering.end());
      
      for (int pIndex = 0; pIndex < phoneOrdering.size(); pIndex++) {
        // access phone id from random ordering of phones
        int phone = phoneOrdering[pIndex];
        
        // if not using TIMIT labels, set modulator signal for the phone
        if (PARAMS->soundType != 3) initModulatorNodes(F1, PARAMS, phone);
        
        // randomize field and settle
        setToRandState(F1);
        
        //choose a random exemplar of the phone to train
        int exemp = floor((rand()*PARAMS->theData[phone].size())/RAND_MAX);
//              int exemp = 0;
        
//        printf("*** size of phone: %d\n", PARAMS->theData[phone].size());
        
        
        if(PARAMS->verbosity == 1)
          printf("trial: %d, phone: %d, exemp: %d\n", pIndex+1, phone+1, exemp+1);
        
        initDriveSig(F1, PARAMS->theData[phone][exemp][0].data());
        
        // tick through the exemplar
        unsigned long numTicks = PARAMS->theData[phone][exemp].size();
        for (int t = 0; t < numTicks; t++){
          
          //-- initialize the driver and modulator signals:
          if(PARAMS->soundType == 1) // processing file
            DNFsetSigs(F1, 1, PARAMS->theData[phone][exemp][t].data());
          //if(PARAMS->soundType == 2) // processing microphone
          //  DNFsetSigs(F1, 1, PARAMS->theData[phone][exemp][t].data());
          else if(PARAMS->soundType == 3) // processing TIMIT
            DNFsetSigs(F1, 2, PARAMS->theData[phone][exemp][t].data());
          else
            printf("invalid soundType\n");
          
          //-- update the field
          DNFrespond(F1);
          
          //-- update the filters
          DNFlearn(F1);
          
          if (PARAMS->viewStyle > 0)
            animation.AnimateField(PARAMS->viewStyle, F1->gDIM, F1->gTPB);
          
        } // end t loop
      } // end phone loop
    } // end cycle loop
    
    if (PARAMS->verbosity == 1) printf("saving DNF training progress\n");
    saveCTC(fileName, F1, PARAMS);
  } // end epoch
  
  return 0;
}


/*************************************************************************
int testDNF(FieldData* F1, const CTCFile* PARAMS) { 

  float* outNodesPtr = (float*) malloc(F1->gModSigSize);

  int phone;
  int node;
  
  for (int torus = 0; torus < PARAMS->numToruses; torus++){

    float err = 0; // accumulate error across exemplars & phones of a field 
    int num_correct = 0;
    int num_incorrect = 0;
    
    // for each mod node (phone) projecting to the torus in question:
    for(int phoneIdx = 0; phoneIdx < PARAMS->theMap[torus].size(); phoneIdx++){
      phone = PARAMS->theMap[torus][phoneIdx];

      // set up target response for phones of the torus for targ phone:
      float targOut[PARAMS->theMap[torus].size()];
      for(int x = 0; x<PARAMS->theMap[torus].size(); x++) targOut[x] = 0;
      targOut[phone] = 1;
      
      // for each exemplar of the phone, accumulate the ouput signal:
      for(int exemp = 0; exemp < PARAMS->theData[phone].size(); exemp++){ 
        if (PARAMS->verbosity == 1) 
          printf("phone: %d, exemp: %d, \n", phone, exemp);
        
        vector<float> torusAcc(PARAMS->theMap[torus].size(), 0);
        
        // tick through the exemplar
        unsigned long numTicks = PARAMS->theData[phone][exemp].size();
        for (int t = 0; t < numTicks; t++){
          DNFsetSigs(F1, 3, PARAMS->theData[phone][exemp][t].data());

          // generate output from the torus
          DNFgenOutput(F1, outNodesPtr, torus);

          // accumulate prediction from the field for specific torus
          for (int nodeIdx = 0; nodeIdx < PARAMS->theMap[torus].size(); nodeIdx++){
            node = PARAMS->theMap[torus][nodeIdx];
            torusAcc[nodeIdx] += ((float) outNodesPtr[node] / numTicks);
          }
        } // end of tick loop
        
        if (PARAMS->verbosity == 1){
          printf("---------------output: \n");
          for(int x = 0; x < PARAMS->theMap[torus].size(); x++){
              printf("%f ", torusAcc[x]);
          }
          printf("\n");
        }

        // --- calculate error for the exemplar
        int bestOut = 0;
        float bestScore = -99999999999;
        float meanScore = 0;
        for (int i = 0; i < PARAMS->theMap[torus].size(); i++) {
          meanScore += torusAcc[i];
          // this isn't very meaningful because of squashing..
          err += abs(targOut[i] - torusAcc[i]);

          // find the winner
          if (torusAcc[i] > bestScore) {
            bestOut = i;
            bestScore = torusAcc[i];
          }
        }
                
        meanScore = ((float) meanScore/PARAMS->theMap[torus].size());
        
        int theWinner = PARAMS->theMap[torus][bestOut];
        if (PARAMS->verbosity == 1){
          printf("top choice: %d (%d/%d) ", 
            theWinner, (int) round(bestScore), (int) round(meanScore));
        }
        if (bestOut == phoneIdx){
            num_correct++;
            if (PARAMS->verbosity == 1) printf("\t\t\t(correct)\n");
        }
        else{
            num_incorrect++;
            if (PARAMS->verbosity == 1) printf("\n");
        }        
      } // end exemp loop
      
      if (PARAMS->verbosity == 1) printf("\n");
    } // end phone loop
    
    float pCorrect = ((float) num_correct/(num_correct+num_incorrect));
    printf("torus %d { ", torus);
    for (int theNode = 0; theNode < PARAMS->theMap[torus].size(); theNode++)
        printf("%d ", PARAMS->theMap[torus][theNode]);
    printf("} - percent correct: %f\n", pCorrect);
  }// end torus loop

  free(outNodesPtr);
  
  return 0;
}


/*************************************************************************
int seeModDNF(FieldData* F1, const CTCFile* PARAMS) { 

  // set up animation window
  CPUAnimBitmap animation(F1->gDIM, F1->gDIM, F1);
  if (PARAMS->viewStyle > 0) { // if viewStyle is non-zero, animating:
    cudaMalloc((void**) &animation.dev_bitmap, animation.image_size());
    animation.initAnimation();
  }

  float pulseAmp[1];
  float freq = 2;     // in Hz
  
  
  while(true){
  
    //create random ordering of recordings
    vector<int> phoneOrdering;
    for (int i = 0; i < F1->gNMN; i++){
      phoneOrdering.push_back(i);
    }
    random_shuffle(phoneOrdering.begin(), phoneOrdering.end());
      
    for (int pIndex = 0; pIndex < phoneOrdering.size(); pIndex++) {
      // access phone id from random ordering of phones
      int phone = phoneOrdering[pIndex];
      
      if (PARAMS->soundType != 3) initModulatorNodes(F1, PARAMS, phone);

      int exemp = floor((rand()*PARAMS->theData[phone].size())/RAND_MAX);
      
      // randomize field and settle
      setToRandState(F1);

      for (int t = 0; t < 100; t++){ // 100 time steps = 1 second
        // pulse amplitude is val of a sin wave at given instant:
        pulseAmp[1] =  (-cos(2.0*M_PI*(t/100.0)*freq)/2.0) + 0.5 ;
        // get the driver and modulator signals:
        DNFsetSigs(F1, 4, pulseAmp);
    
        // update the field
        DNFrespond(F1);

        if (PARAMS->viewStyle > 0) 
          animation.AnimateField(PARAMS->viewStyle, F1->gDIM, F1->gTPB);
      
      } // end tick
    } // end pIndex
  } // end while
  
  return 0;
}


/*************************************************************************
int seeDriveDNF(FieldData* F1, const CTCFile* PARAMS) { 

  // set up animation window
  CPUAnimBitmap animation(F1->gDIM, F1->gDIM, F1);
  if (PARAMS->viewStyle > 0) { // if viewStyle is non-zero, animating:
    cudaMalloc((void**) &animation.dev_bitmap, animation.image_size());
    animation.initAnimation();
  }
  
  for (int phone = 0; phone < F1->gNMN; phone++) {
    for (int exemp = 0; exemp < 1; exemp++){
      
      // randomize field and settle
      setToRandState(F1);

      initDriveSig(F1, PARAMS->theData[phone][exemp][0].data());
      // tick through the exemplar
      unsigned long numTicks = PARAMS->theData[phone][exemp].size();
      for (int t = 0; t < numTicks; t++){

        // get the driver and modulator signals:
        DNFsetSigs(F1, 3, PARAMS->theData[phone][exemp][t].data());
    
        // update the field
        DNFrespond(F1);

        if (PARAMS->viewStyle > 0) 
          animation.AnimateField(PARAMS->viewStyle, F1->gDIM, F1->gTPB);
      
      } // end tick
    } // end exemplar
  } // end phone
  
  return 0;
}

/*************************************************************************
// view both drive an mod with learning turned off
int seeBothDNF(FieldData* F1, const CTCFile* PARAMS) { 

  // set up animation window
  CPUAnimBitmap animation(F1->gDIM, F1->gDIM, F1);
  if (PARAMS->viewStyle > 0) { // if viewStyle is non-zero, animating:
    cudaMalloc((void**) &animation.dev_bitmap, animation.image_size());
    animation.initAnimation();
  }
  
  for (int phone = 0; phone < F1->gNMN; phone++) {
    // if not using TIMIT labels, set modulator signal for the phone
    if (PARAMS->soundType != 3) initModulatorNodes(F1, PARAMS, phone);

    for (int exemp = 0; exemp < 1; exemp++){
      
      // randomize field and settle
      setToRandState(F1);

      initDriveSig(F1, PARAMS->theData[phone][exemp][0].data());
      // tick through the exemplar
      unsigned long numTicks = PARAMS->theData[phone][exemp].size();
      for (int t = 0; t < numTicks; t++){

        // get the driver and modulator signals:
        DNFsetSigs(F1, 1, PARAMS->theData[phone][exemp][t].data());
    
        // update the field
        DNFrespond(F1);

        if (PARAMS->viewStyle > 0) 
          animation.AnimateField(PARAMS->viewStyle, F1->gDIM, F1->gTPB);
      
      } // end tick
    } // end exemplar
  } // end phone

  return 0;
}

/*************************************************************************
int testSound(FieldData* F1, const CTCFile* PARAMS) {
  
  for (int phone = 0; phone < F1->gNMN; phone++){
    for (int exemp = 0; exemp<PARAMS->theData[phone].size(); exemp++){

      // tick through the exemplar
      for (int t = 0; t < PARAMS->theData[phone][exemp].size(); t++){

        for (int coeff = 0; coeff<F1->gNDN; coeff++){
            printf("%f ", PARAMS->theData[phone][exemp][t][coeff]);
        }
        printf("\n");
      }
    }
  }
    
  return 0;
}


/*************************************************************************
int testTimeWarpNFCs(FieldData* F1, const CTCFile* PARAMS) {

  vector<vector<vector<vector<double>>>> CTCprocessed;

  for (int phone = 0; phone < PARAMS->theData.size(); phone++) {
    vector<vector<vector<double>>> CTCPhone;
    for (int exemp = 0; exemp < PARAMS->theData[phone].size(); exemp++) {

      //initDriveSig(F1, initSig.data());
      
      initDriveSig(F1, PARAMS->theData[phone][exemp][0].data());
      setToDefaultState(F1);
      // tick through the exemplar
      printf("%d %d\n", phone, exemp);
//      std::cerr << phone << " " << exemp << "  "<<endl;
      vector<vector<double >> CTCexemp;
      for (int t = 0; t < PARAMS->theData[phone][exemp].size(); t++) {

        float* modNodesPtr = (float*) malloc(F1->gModSigSize);
  

//      DNFtick(F1, PARAMS, modNodesPtr, PARAMS->theData[phone][exemp][t].data());

        //fix this allocation
        vector<double> temp(modNodesPtr, modNodesPtr + F1->gNMN);
        vector<double> CTCTick = temp;
        CTCexemp.push_back(CTCTick);

        free(modNodesPtr);
      }
      CTCPhone.push_back(CTCexemp);
    } // end exemp loop
    //cout << endl;
    CTCprocessed.push_back(CTCPhone);
  } // end phone loop

  sound.data.processed = CTCprocessed;
  //sr.data.stats();
  //sr.prototypes.buildAllPrototypes(sr.data);
  //sr.calibration.leaveOneOutTest(sr.data, sr.prototypes, 30);
  sound.calibration.classify(sound.data);
  return 0;
}


/*************************************************************************
int testTimeWarpMFCCs(char* soundDirName, const CTCFile* PARAMS) {

  Asr sound(1);
 
  if(PARAMS->preProcess == 1) // 1 for load, 2 for import
    sound.input->loadFiles(soundDirName);  // for .wav files (C++ MFCCs)
  else
    sound.input->import(soundDirName);   // for .txt files (matlab MFCCs)

  sound.calibration.classify(sound.data);
  
  return 0;
}

/*************************************************************************
// this version tests the success of individual fields
int generateERPs(FieldData* F1, const CTCFile* PARAMS){ 
  
  float* modNodesPtr = (float*) malloc(F1->gModSigSize);
  
  // allocate a memory buffer for saving field info:
	float* memBuff1 = (float*) malloc(F1->gFieldSize);
  float* memBuff2 = (float*) malloc(F1->gFieldSize);
  float raw;
  float change;
  float driveSig;
  float driveSigErr;
  float modSig;
  float modSigErr;
  float driveModDiff;
  
  for (int phone = 0; phone < F1->gNMN; phone++) {
    initModulatorNodes(F1, PARAMS, phone);
    
    for (int exemp = 0; exemp < PARAMS->theData[phone].size(); exemp++){
      
      initDriveSig(F1, PARAMS->theData[phone][exemp][0].data());

      // THIS IS ONE OF THE BIG QUESTIONS, WHAT STATE SHOULD IT BE IN?
//      setToDefaultState(F1);
      // tick through the exemplar
      unsigned long numTicks = PARAMS->theData[phone][exemp].size();
      for (int t = 0; t < numTicks; t++){
        
        // type 1 below sets driver signal and mod signal from file
        DNFsetSigs(F1, 1, PARAMS->theData[phone][exemp][t].data());

        // update the field 
        DNFrespond(F1);
        
        // print EEG signals
        // save absolute change in field
        cudaMemcpy(memBuff1, F1->change, F1->gFieldSize, cD2H);
        change = 0;
        for (int cnt = 0; cnt < F1->gDIM * F1->gDIM; cnt++){
          change += abs(memBuff1[cnt]);
        }

        // save raw field activation
        cudaMemcpy(memBuff1, F1->raw, F1->gFieldSize, cD2H);
        raw = 0;
        for (int cnt = 0; cnt < F1->gDIM * F1->gDIM; cnt++){
          raw += memBuff1[cnt];
        }
        
        // save modSig & modSigErr
        cudaMemcpy(memBuff2, F1->modSig, F1->gFieldSize, cD2H);
        modSig = 0;
        modSigErr = 0;
        for (int cnt = 0; cnt < F1->gDIM * F1->gDIM; cnt++){
          modSig += memBuff2[cnt];
          modSigErr += abs(memBuff2[cnt] - memBuff1[cnt]);
        }

        // save driveSig & driveSigErr
        cudaMemcpy(memBuff2, F1->driveSig, F1->gFieldSize, cD2H);
        driveSig = 0;
        driveSigErr = 0;
        for (int cnt = 0; cnt < F1->gDIM * F1->gDIM; cnt++){
          driveSig += memBuff2[cnt];
          driveSigErr += abs(memBuff2[cnt] - memBuff1[cnt]);
        }

        // save diff between drive sig and mod sig
        cudaMemcpy(memBuff1, F1->modSig, F1->gFieldSize, cD2H);
        driveModDiff = 0;
        for (int cnt = 0; cnt < F1->gDIM * F1->gDIM; cnt++){
          driveModDiff += abs(memBuff2[cnt] - memBuff1[cnt]);
        }
        
        printf("%f %f %f %f %f %f %f\n", 
          raw, change, modSig, modSigErr, driveSig, driveSigErr, driveModDiff);
        
        // COULD ALSO USE: 1) SUM OF WEIGHTS CHANGE, 2) CHANGE IN DRIVE SIG, 
        // 3) CHANGE IN MOD SIG, 4) DIFF BETWEEN CHANGE IN DRIVE SIG AND 
        // CHANGE IN MOD SIG, 5) CHANGE IN FIELD DIVIDED BY RAW OF FIELD, ETC..
        
      }
    }
  }
  
  free(memBuff1);
  free(memBuff2);
  
  return 0;
}

/*************************************************************************/






/**************************************************************************
                       SAVING AND LOADING .CTC FILES
**************************************************************************
// (overwrites any existing file)
int saveCTC(char* fileName, FieldData* F, const CTCFile* PARAMS){

if (PARAMS->verbosity == 1) printf("writing to: %s\n", fileName);

// open the file
FILE *outfp;
if((outfp = fopen(fileName, "w+b")) == NULL){
	printf("can't open: %s\n", fileName);
	return 1;
}

// copy drive weights from device to host
unsigned long driveFiltSize = F->gDriveSigSize * F->gDIM * F->gDIM;
float* dweights = (float*) malloc(driveFiltSize);
cudaMemcpy(dweights, F->dFilt, driveFiltSize, cD2H);

// copy mod weights from device to host
unsigned long modFiltSize = F->gModSigSize * F->gDIM * F->gDIM;
float* mweights = (float*) malloc(modFiltSize);
cudaMemcpy(mweights, F->mFilt, modFiltSize, cD2H);

// copy out weights from device to host (same size as mod weights)
//float* oweights = (float*) malloc(modFiltSize);
//cudaMemcpy(oweights, F->oFilt, modFiltSize, cD2H);

// copy default state from device to host
float* defaultState = (float*) malloc(F->gFieldSize);
cudaMemcpy(defaultState, F->dState, F->gFieldSize, cD2H);

// FULL FILE SHOULD HAVE A HEADER WITH ADDRESSES OF CHUNKS
// AND A CHUNK WITH PARAMS INFORMATION
// write header - size file, num chunks, size of each chunk
// write comm chunk - params (chunk 1)

// write driver weights (chunk 2)
unsigned long numBytesWritten;
numBytesWritten = fwrite(dweights, 1, driveFiltSize, outfp);
if (PARAMS->verbosity == 1) printf("num driver weights written: %d\n", 
                                            numBytesWritten/sizeof(float));

// write mod weights (chunk 3)
numBytesWritten = fwrite(mweights, 1, modFiltSize, outfp);
if (PARAMS->verbosity == 1) printf("num mod weights written: %d\n", 
                                            numBytesWritten/sizeof(float));

// write out weights (chunk 4)
//numBytesWritten = fwrite(oweights, 1, modFiltSize, outfp);
//if (PARAMS->verbosity == 1) printf("num out weights written: %d\n", 
//                                            numBytesWritten/sizeof(float));

// write default state (chunk 5)
numBytesWritten = fwrite(defaultState, 1, F->gFieldSize, outfp);
if (PARAMS->verbosity == 1) printf("num default state units written: %d\n", 
                                            numBytesWritten/sizeof(float));

free(dweights);
free(mweights);
//free(oweights);
free(defaultState);

return 0;

}


/*************************************************************************
int loadCTC(char* fileName, FieldData* F, const CTCFile* PARAMS){

if (PARAMS->verbosity == 1) printf("loading from: %s\n", fileName);

// open the file
FILE *infp;
if((infp = fopen(fileName, "r+b")) == NULL){
	printf("can't open: %s\n", fileName);
	return 1;
}
unsigned long numBytesRead;


unsigned long driveFiltSize = F->gDriveSigSize * F->gDIM * F->gDIM;
float* dweights = (float*) malloc(driveFiltSize);
numBytesRead = fread(dweights, 1, driveFiltSize, infp);
if (PARAMS->verbosity == 1) printf("num driver weights read: %d\n", 
                                               numBytesRead/sizeof(float));

unsigned long modFiltSize = F->gModSigSize * F->gDIM * F->gDIM;
float* mweights = (float*) malloc(modFiltSize);
numBytesRead = fread(mweights, 1, modFiltSize, infp);
if (PARAMS->verbosity == 1) printf("num mod weights read: %d\n", 
                                               numBytesRead/sizeof(float));

//float* oweights = (float*) malloc(modFiltSize);
//numBytesRead = fread(oweights, 1, modFiltSize, infp);
//if (PARAMS->verbosity == 1) printf("num out weights read: %d\n", 
//                                               numBytesRead/sizeof(float));

float* defaultState = (float*) malloc(F->gFieldSize);
numBytesRead = fread(defaultState, 1, F->gFieldSize, infp);
if (PARAMS->verbosity == 1) printf("num default state units read: %d\n", 
                                               numBytesRead/sizeof(float));

cudaMemcpy(F->dFilt, dweights, driveFiltSize, cH2D);
cudaMemcpy(F->mFilt, mweights, modFiltSize, cH2D);
//cudaMemcpy(F->oFilt, oweights, modFiltSize, cH2D);
cudaMemcpy(F->dState, defaultState, F->gFieldSize, cH2D);

free(dweights);
free(mweights);
//free(oweights);
free(defaultState);

return 0;

}


/**************************************************************************
                       INTERFACE HELPER FUNCTIONS
**************************************************************************

int setDefaults(CTCFile *PARAMS){

    PARAMS->verbosity   = 0;
    PARAMS->runMode     = 1;   
    return 0;     
}


/**************************************************************************
                       INTERFACE HELPER FUNCTIONS
**************************************************************************

int setDefaults(CTCFile *PARAMS){

    PARAMS->verbosity   = 0;
    PARAMS->runMode     = 1;        
    PARAMS->viewStyle   = 0;        
    PARAMS->soundType   = 1;        // 1=file, 2=microphone, 3 = TIMIT
//    PARAMS->absorption  = 0;      // absorption style processing (?)
    PARAMS->initState   = 0;        // 0=rand, 1=default, 2=previous 
    PARAMS->styleInput  = 1;        // 1 = all mfccs (root + diffs = 26), 
                                    // 2 = root (13), 3 = diffs (13)
    
    PARAMS->preProcess = 1;         // 1 = C (load), 2 = Matlab (import)
    
    PARAMS->gridWidth   = 4;       // e.g. 4*4 = 16 toruses
//    PARAMS->torusWidth  = 32;     // number of units per dim of torus - hard wired at 32
//    PARAMS->nDriveNodes = 0;        // num MFCC coeffs - set {0,   1,    2};later
//    PARAMS->nModNodes   = 0;        // num output words - set later
    
    PARAMS->noise       = 1.0;      // noise
//    PARAMS->dLearnRate  = 0.0001;     // 
    PARAMS->dLearnRate  = 0.01;     // with exponential on
    PARAMS->mLearnRate  = 0.001;    // with exponential on
    PARAMS->rWin        = 10;       // num time steps for the running avg of a field
    
//    PARAMS->dGain       = 15.0;     // f
    PARAMS->dGain       = 15.0;     // 30 // drive gain
    PARAMS->mGain       = 15.0;     // mod gain
    PARAMS->fGain       = 1.0;      // full gain
    PARAMS->pGain       = 0.1;      // perturb gain
    PARAMS->sGain       = 1.0;      // shunting gain

    PARAMS->wLimit      = 500;      // wLimit
    PARAMS->settleTime  = 20;       // settleTime
    PARAMS->nItersPerEpoch = 20;     // number of training cycles per epoch
    PARAMS->nTrainEpochs  = 1;     // num times to stop and save weights 
            
    return 0;
}

/*************************************************************************/
// THIS IS LACKING OUTDATED - CLEAN UP SOME DAY!
int usage()
{
    printf("USAGE:\n");
   
	return(0);
}

/*************************************************************************
int viewParams(const CTCFile *PARAMS){
    
//    char string[80];
  
    int theErr = 0;
    
    printf("--- PARAMETERS: ---\n");

//    printf("filename: %s\n", PARAMS->filename);
    printf("run mode: %d\n", PARAMS->runMode); 
    
    printf("display mode: %d\n", PARAMS->viewStyle);

    printf("sound type: %d\n", PARAMS->soundType); 
    
    switch(PARAMS->initState){
    	case 0 : printf("init state type: random\n"); break; 
    	case 1 : printf("init state type: default\n"); break; 
      case 2 : printf("init state: previous\n"); break; 
      default : printf("init state: ERROR \n"); theErr++; break;
    }
    
    printf("grid width: %d\n", PARAMS->gridWidth);
//    printf("torus width: %d\n", PARAMS->torusWidth);
    printf("num drive nodes: %d\n", PARAMS->nDriveNodes);
    printf("num mod nodes: %d\n", PARAMS->nModNodes);
    printf("num toruses: %d \n", PARAMS->numToruses);
    
    printf("noise: %f\n", PARAMS->noise);
    printf("driver learn rate: %f\n", PARAMS->dLearnRate);
    printf("modulator learn rate: %f\n", PARAMS->mLearnRate);
    printf("run avg window size: %d\n", PARAMS->rWin);
    printf("drive gain: %f\n", PARAMS->dGain);
    printf("mod gain: %f\n", PARAMS->mGain);
    printf("full gain: %f\n", PARAMS->fGain);
    printf("perturbation gain: %f\n", PARAMS->pGain);
    printf("shunting gain: %f\n", PARAMS->sGain);
 
    printf("weight growth limiter: %d\n", PARAMS->wLimit);
    printf("settle time: %d\n", PARAMS->settleTime);
    printf("num iterations per training epoch: %d\n", PARAMS->nItersPerEpoch);

    return theErr;
}

/*************************************************************************/
char crack(int argc, char** argv, char* flags, int ignore_unknowns)
{
    char *pv, *flgp;

    while ((arg_index) < argc){
        if (pvcon != NULL)	// more than 1 flag after "-"
            pv = pvcon;		// pv continued
        else{
            if (++arg_index >= argc) return(NULL); 
            pv = argv[arg_index];
            if (*pv != '-') 
                return(NULL);
            }
        pv++;		// skip '-' or prev. flag

        if (*pv != NULL){
            if ((flgp=strchr(flags,*pv)) != NULL){ 	// if valid flag
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


