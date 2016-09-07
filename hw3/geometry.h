#ifndef hGeometryLib
#define hGeometryLib

#include <stdbool.h>
#include "midiparse.h"



typedef struct Point {
    double x, y;
} Point;

typedef struct Color {
    float r, g, b;
} Color;

typedef struct Circle {
    Point center;
    int radius;
    bool fill;
} Circle;

/*typedef struct Rectangle {
    Point ur;
    Point ul;
    Point lr;
    Point ll;
    bool fill;
} Rectangle;*/

typedef union Shape {
    Circle circle;
    //Rectangle rectangle;
} Shape;

typedef enum Shape_Type {
    CIRCLE //,
    //RECTANGLE
} Shape_Type;


typedef struct Properties {
    //double growth_rate; //amount by which shape expands or contracts 
    double      momentumX;
    double      momentumY;

    double      max_radius;
    double      min_radius;

    int         timeSinceLastChanged;
    int         growthRate; // either positive or negative
    SoundEvent  lastSoundEvent;


} Momentum;


typedef struct Geometry {
    Shape_Type type;
    Shape shape; 
    Color color;
    int thickness;
    Properties properties;
    bool displayOn;
} Geometry;




#endif

