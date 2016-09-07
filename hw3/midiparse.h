#ifndef hMidiParseLib
#define hMidiParseLib

typedef struct SoundEvent {
    int pitch, volume;
} SoundEvent;

typedef struct DiscreteTracks {
    int trackLength;
    int samplesPerSecond;
    int numTracks;
    SoundEvent **tracks;
} DiscreteTracks;

void printDiscreteTracks(DiscreteTracks dts);
void printSoundEvent(SoundEvent s);

DiscreteTracks getDiscreteTracks(char* fileName, int samplesPerSecond);
void freeDiscreteTracks(DiscreteTracks dts);

#endif