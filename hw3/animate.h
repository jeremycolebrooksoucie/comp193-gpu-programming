#include <GL/freeglut.h>
#include <GL/glext.h>
#include <GL/glx.h>

#include "gpu_main.h"

//#include <iostream>

struct CPUAnimBitmap {
    unsigned char* pixels;
    unsigned char* dev_bitmap;

    int gWidth, gHeight;
    int print_width, print_height;
    void (* clickDrag)(void*, int, int, int, int);
    int dragStartX, dragStartY;

    AParams *params;
    GPU_Palette* thePalette;

    CPUAnimBitmap(AParams *params, GPU_Palette* d); 
    
    ~CPUAnimBitmap();

    unsigned char* get_ptr(void) const { return pixels; }

    long gImage_size(void) const { return gWidth * gHeight * 4 * sizeof(char); }
    long print_image_size(void) const {return print_width * print_height * 4* sizeof(char);}
    void click_drag(void (* f)(void*, int, int, int, int));

    //void anim_and_exit(void (* f)(void*, int), void(* e)(void*));
    void anim_and_exit(void (* f)(void*), void(* e)(void*));

    // static method used for glut callbacks
    static CPUAnimBitmap** get_bitmap_ptr(void);

    // static method used for glut callbacks
    static void mouse_func(int button, int state, int mx, int my);

    // static method used for glut callbacks
    static void idle_func(void);

    // static method used for glut callbacks
    static void Key(unsigned char key, int x, int y);
    // static method used for glut callbacks
    static void Draw(void);

    void initAnimation();
    void drawPalette();
};


__global__ void drawGray(unsigned char* optr, const float* gray);
__global__ void drawColor(unsigned char* optr,
                          const float* red,
                          const float* green,
                          const float* blue);
