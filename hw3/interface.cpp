#include <stdio.h>
#include <ctype.h>
#include <cstdlib>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>


#include "interface.h"
#include "gpu_main.h"
#include "params.h"
#include "animate.h"
#include "qdbmp.h"
#include "geometry.h"

int main(int argc, char *argv[]) {

    // declare params and default behavior
    AParams params;
    params = getDefaultParameters();

    // TODO: usage

    // parses commandline arguments
    opterr = 0;
    int c;
    while ((c = getopt (argc, argv, "avb:m:s:o:")) != -1) {
        switch (c)
        {
        case 'a': //after image
            params.afterimage = true;
            break;
        case 'b': //image
            params.bmpFile = optarg;
            break;
        case 'm': // midi
            params.midiFile = optarg;
            break;
        case 's': // sampling rate
            params.samplingRate = atoi(optarg); //TODO: unsafe string handling
            break;
        case 'v': // verbose mode
            params.verbosity = true;
            break;
        case '?':
            if (optopt == 'b' || optopt == 'm' || optopt == 's')
                fprintf (stderr, "Option -%c requires an argument.\n", optopt);
            else if (isprint (optopt))
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf (stderr,
                         "Unknown option character `\\x%x'.\n",
                         optopt);
            return 1;
        default:
            abort ();
        }
    }

    // initialize memory on the gpu
    GPU_Palette gpuPallete;
    GPU_Geometries gpuGeometries;
    GPU_Midi gpuMidi;

    // read in the image file, write the data to the gpu:
    int err1 = openBMP(&params, &gpuPallete);
    int err3 = openMidi(&params, &gpuMidi);
    int err2 = initGeometries(&gpuGeometries, params, gpuMidi);

    if (params.verbosity) 
        viewParams(params);
    

    if (params.verbosity)
        printf("\n -- RUNNING ON GPU -- \n");

    runIt(gpuPallete, gpuGeometries, gpuMidi, params);


    // TODO: clean up memory here

}



/**************************************************************************
                    PROGRAM FUNCTIONS
**************************************************************************/

int runIt(GPU_Palette gpuPallete, GPU_Geometries gpuGeometries, GPU_Midi gpuMidi, AParams params) {

    printf("runit \n");
    CPUAnimBitmap animation(&params, &gpuPallete);
    printf("animiation instantion successful\n");

    animation.initAnimation();

    while (params.curFrame < params.duration) {
        if (params.verbosity)
            fprintf(stderr, "Iteration: %d \n", params.curFrame);
        updateGeometries(gpuPallete, gpuGeometries, gpuMidi, params);

        drawGeometries  (gpuPallete, gpuGeometries, gpuMidi, params);
        animation.drawPalette();

        params.curFrame++;
    }

    return 0;
}

/**************************************************************************
                                             PROGRAM FUNCTIONS
**************************************************************************/



/* Allocates array of Geometry in Cuda */
int initGeometries(GPU_Geometries *gpuPallete, const AParams params, const GPU_Midi gpuMidi) 
{

    int numGeometries;
    numGeometries = gpuMidi.numTracks;



    Geometry *geo_list = new Geometry [numGeometries];
    for (int i = 0; i < numGeometries; i++)
        geo_list[i] = randomGeometry(params);

    (*gpuPallete) = initGPUGeometries(numGeometries, geo_list);

    free(geo_list);
    return 0;
}

Geometry randomGeometry(AParams params)
{
    Geometry geometry;
    int x, y, radius;
    x = rand() % params.print_width;
    y = rand() % params.print_height;
    radius = 0; //(rand() % params.print_width % params.print_height)/4;

    geometry.type = CIRCLE;
    Circle c = {.center = newPoint(x, y),
                .radius = radius,
                .fill = rand() % 2
               };
    geometry.shape.circle = c;

    float r, g, b;
    r = ((double) (rand() % 255)) / 255;
    g = ((double) (rand() % 255)) / 255;
    b = ((double) (rand() % 255)) / 255;

    geometry.color = newColor(r, g, b);

    // randomly allocate 
    int thickness;
    double momentumX, momentumY;
    double min_radius, max_radius;
    int timeSinceLastChanged;
    int growthRate;
    SoundEvent soundEvent;

    thickness = rand() % 10 + 1;
    momentumX = rand() % 6;
    momentumY = rand() % 6;
    min_radius = rand() % 40 + 10;
    max_radius = min_radius + rand() % 50 + 10;
    timeSinceLastChanged = 0;
    //if(rand() % 2 == 1)
    growthRate = 1;
    //else
    //    growthRate = -1;



    soundEvent = {.pitch = 0, .volume = 0};
    geometry.thickness = thickness;
    Properties prop = {.momentumX   = momentumX,  .momentumY  = momentumY, 
                        .max_radius = max_radius, .min_radius = min_radius,
                        .timeSinceLastChanged = timeSinceLastChanged, 
                        .growthRate = growthRate, 
                        .lastSoundEvent = soundEvent };
    geometry.properties = prop;
    geometry.displayOn = true;
    return geometry;
}


/* Allocates sound events in cuda */
int openMidi(AParams *p, GPU_Midi *m)
{
    // allocate discrete tracks

    if (p -> midiFile == NULL) {
        fprintf(stderr, "Please supply midi file \n");
        exit(0);
    }
    DiscreteTracks dts;
    dts = getDiscreteTracks(p -> midiFile, p -> samplingRate);
    p -> duration = dts.trackLength;
    // copy data from dts to gpu
    (*m) = initGPUMidi(dts);

    //printDiscreteTracks(dts);
    freeDiscreteTracks(dts);
    return 0;
}



/*
 * Function loads image into gpuPallete from either bitmap or default all black image
 * Also instantiates the size parameters in params
 */
 // TODO: way too long
int openBMP(AParams* params, GPU_Palette* gpuPallete) 
{
    float *graymap, *redmap, *greenmap, *bluemap;
    if(params -> bmpFile != NULL) {
        BMP* bmp;

        bmp = BMP_ReadFile(params -> bmpFile);
        BMP_CHECK_ERROR( stderr, -1 );

        params->print_width = BMP_GetWidth( bmp );
        params->print_height = BMP_GetHeight( bmp );
        // use integer mult and div to round to nearest multiple of THREADS_PER_BLOCK
        params->width =  ((params->print_width  + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * THREADS_PER_BLOCK;
        params->height = ((params->print_height + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * THREADS_PER_BLOCK;

        // TODO: unclear if these are needed
        params->size = 3 * params->width * params->height;
        params->print_size = 3 * params->print_width * params->print_height;


        (*gpuPallete) = initGPUPalette(params);

        // need to use calloc to have 'extra' cells be empty, this requires
        // some manipulations of parameters
        graymap = (float*) calloc(gpuPallete->gSize / sizeof(float), sizeof(float));
        redmap = (float*) calloc(gpuPallete->gSize / sizeof(float), sizeof(float));
        greenmap = (float*) calloc(gpuPallete->gSize / sizeof(float), sizeof(float));
        bluemap = (float*) calloc(gpuPallete->gSize / sizeof(float), sizeof(float));



        for (int x = 0; x < params -> print_width; x++)
        {
            for (int y = 0; y < params -> print_height; y++)
            {
                unsigned char r, g, b;
                int tmp_y = params -> print_height - (y + 1);//needed to not invert image
                BMP_GetPixelRGB(bmp, x, tmp_y, &r, &g, &b);

                // segregate data as floats between 0 - 1 to be written to gpu
                int graymapIdx = x + y * params -> width;
                graymap[graymapIdx]   = (float) (r + g + b) / (255.0 * 3.0);
                redmap[graymapIdx]    = (float) r / 255.0;
                greenmap[graymapIdx]  = (float) g / 255.0;
                bluemap[graymapIdx]   = (float) b / 255.0;
            }
        }
        BMP_Free( bmp );
    } 
    /* Case for no image file provided */
    else {
        params->print_width = 400;
        params->print_height = 400;
        // use integer mult and div to round to nearest multiple of THREADS_PER_BLOCK
        params->width =  ((params->print_width  + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * THREADS_PER_BLOCK;
        params->height = ((params->print_height + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * THREADS_PER_BLOCK;

        // TODO: unclear if these are needed
        params->size = 3 * params->width * params->height;
        params->print_size = 3 * params->print_width * params->print_height;

        (*gpuPallete) = initGPUPalette(params);

        // need to use calloc to have 'extra' cells be empty, this requires
        // some manipulations of parameters
        graymap = (float*) calloc(gpuPallete->gSize / sizeof(float), sizeof(float));
        redmap = (float*) calloc(gpuPallete->gSize / sizeof(float), sizeof(float));
        greenmap = (float*) calloc(gpuPallete->gSize / sizeof(float), sizeof(float));
        bluemap = (float*) calloc(gpuPallete->gSize / sizeof(float), sizeof(float));



        for (int x = 0; x < params -> print_width; x++)
        {
            for (int y = 0; y < params -> print_height; y++)
            {
                unsigned char r, g, b;
                r = g = b = 0;

                // segregate data as floats between 0 - 1 to be written to gpu
                int graymapIdx = x + y * params -> width;
                graymap[graymapIdx]   = (float) (r + g + b) / (255.0 * 3.0);
                redmap[graymapIdx]    = (float) r / 255.0;
                greenmap[graymapIdx]  = (float) g / 255.0;
                bluemap[graymapIdx]   = (float) b / 255.0;
            }
        }
    }

    // write image data to the GPU
    cudaMemcpy(gpuPallete->gray, graymap, gpuPallete->gSize, cH2D);
    cudaMemcpy(gpuPallete->red, redmap, gpuPallete->gSize, cH2D);
    cudaMemcpy(gpuPallete->green, greenmap, gpuPallete->gSize, cH2D);
    cudaMemcpy(gpuPallete->blue, bluemap, gpuPallete->gSize, cH2D);

    cudaMemcpy(gpuPallete->gray_background, graymap, gpuPallete->gSize, cH2D);
    cudaMemcpy(gpuPallete->red_background, redmap, gpuPallete->gSize, cH2D);
    cudaMemcpy(gpuPallete->green_background, greenmap, gpuPallete->gSize, cH2D);
    cudaMemcpy(gpuPallete->blue_background, bluemap, gpuPallete->gSize, cH2D);

    // free CPU memory
    free(graymap);
    free(redmap);
    free(greenmap);
    free(bluemap);

    return 0;
}





/**************************************************************************
                    PARAMETER HELPER FUNCTIONS
**************************************************************************/
AParams getDefaultParameters() {

    AParams params;
    params.verbosity = false;
    params.samplingRate = 3;
    params.duration = 0; // should get set by file length
    params.midiFile = NULL;
    params.bmpFile = NULL;
    params.afterimage = false;

    return params;
}

void viewParams(AParams params) {

    printf("--- PARAMETERS: ---\n");

    fprintf(stderr, "midi filename \t : \t %s \n", params.midiFile);
    fprintf(stderr, "bmp  filename \t : \t %s \n", params.bmpFile);
    fprintf(stderr, "sampling rate \t : \t %d \n", params.samplingRate);
    fprintf(stderr, "verbosity     \t : \t %d \n", params.verbosity);
    fprintf(stderr, "duration      \t : \t %d \n", params.duration);


}



/**********************************************************************
                         Helpers for Creating Structs               
 **********************************************************************/

Point newPoint(int x, int y)
{
    Point p = {.x = x, .y = y};
    return p;
}

Color newColor(float r, float g, float b)
{
    Color c = {.r = r, .g = g, .b = b};
    return c;
}