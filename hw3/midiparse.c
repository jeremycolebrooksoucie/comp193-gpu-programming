#include "midiparse.h"
#include <stdio.h>
#include "MidiFile.h"
#include "Options.h"
#include <iostream>
#include <iomanip>
#include <tgmath.h>

using namespace std;




SoundEvent* processEvent(MidiEvent* start, MidiEvent* end, SoundEvent* discreteTrack, 
                         int discreteTrackLength, int samplesPerSecond);
SoundEvent* discretizeTrack(MidiFile midifile, int trackID, int numSamples, int samplesPerSecond);

int getDiscreteIndex (double seconds, int samplesPerSecond);
double getLengthInSeconds(MidiFile midifile);

void printDiscreteTracks(SoundEvent **tracks, const int numTracks, const int trackLength);
void printSoundEvent(SoundEvent s);

/*
int main(int argc, char** argv) {

    // note unresilient testing code
    char *fileName;
    fileName = argv[1];

    int samplesPerSecond; 
    samplesPerSecond = stoi(argv[2]);


    DiscreteTracks dts = getDiscreteTracks(fileName, samplesPerSecond);
    printDiscreteTracks(dts);

}
*/

// discritizes midifile at fileName at samplesPerSecond
DiscreteTracks getDiscreteTracks(char* fileName, int samplesPerSecond)
{
    // open midi
    MidiFile midifile;
    midifile.read(fileName);

    // calculate seconds value for each midi event
    midifile.doTimeAnalysis();

    // figure out size of discrete track array based on lenght of track and sampling rate
    double lengthSeconds;
    int tracks, numSamples;
    lengthSeconds   = getLengthInSeconds(midifile) ;
    tracks          = midifile.getTrackCount();
    numSamples      = ceil(lengthSeconds * samplesPerSecond);

    // allocate and fill memory for 2d array of soundEvents 
    SoundEvent** discreteTrackList = new SoundEvent*[tracks];
    for (int i = 0; i < tracks; i++){
        //cout << "Track: " << i << endl;
        discreteTrackList[i] = discretizeTrack(midifile, i, numSamples, samplesPerSecond);
    }

    // fill struct for return
    DiscreteTracks dts = {
        .trackLength = numSamples, 
        .samplesPerSecond = samplesPerSecond,
        .numTracks = tracks, 
        .tracks = discreteTrackList
    };
    return dts;
}




// note: requires linkNotePairs to be called on midifile
// returns dynamically allocated array of 
SoundEvent* discretizeTrack(MidiFile midifile, int trackID, int numSamples, int samplesPerSecond)
{
    // allocate array of soundEvents and fill it with "empty" sounds
    SoundEvent* discreteTrack;
    discreteTrack = new SoundEvent[numSamples];
    SoundEvent empty = {.pitch = 0, .volume = 0};
    for (int i = 0; i < numSamples; i++)
        discreteTrack[i] = empty;

    // gets current raw track and links its note (associates start midi event with end midi event)
    MidiEventList curTrack = midifile[trackID];
    curTrack.linkNotePairs();

    // iterate through all raw events in track and process
    int numEventsRaw = curTrack.size();
    for (int curEvent = 0; curEvent < numEventsRaw; curEvent++) {
        // get pointers to start event and corresponding linked event
        MidiEvent *startEvent, *endEvent;
        startEvent = &curTrack[curEvent];
        endEvent = startEvent -> getLinkedEvent();
        // fill discrete track with appropriate soundEvent between both events
        discreteTrack = processEvent(startEvent, endEvent, discreteTrack, numSamples, samplesPerSecond);
    }

    return discreteTrack;
}

//parses start/end midievents and inserts their soundEvent at appropriate indices into 
// discrete track
SoundEvent* processEvent(MidiEvent* start, MidiEvent* end, SoundEvent* discreteTrack, 
                         int discreteTrackLength, int samplesPerSecond)
{
    // decoding midiEvent, yeah... this is a weird convention
    SoundEvent currentSoundEvent;
    currentSoundEvent.pitch = (*start)[1];
    currentSoundEvent.volume = (*start)[2];

    // getting start and end indices
    int startIndex, endIndex;
    startIndex = getDiscreteIndex(start -> seconds, samplesPerSecond);
    if (end == NULL) 
        endIndex = discreteTrackLength;
    else 
        endIndex = getDiscreteIndex(end -> seconds, samplesPerSecond);

    // set appropriate values n discreteTrack
    for (int i = startIndex; i < endIndex && i < discreteTrackLength; i++)
        discreteTrack[i] = currentSoundEvent;

    return discreteTrack;
}

// figures out appropriate discrete index given second position of event and sampling rate
int getDiscreteIndex (double seconds, int samplesPerSecond)
{
    return ceil(seconds * samplesPerSecond);
}

// gets the length of a midifile
double getLengthInSeconds(MidiFile midifile) {
    midifile.joinTracks();
    int lastEventIndex = midifile[0].size() - 1; 
    double lastTime = midifile[0][lastEventIndex].seconds;
    midifile.splitTracks();

    return lastTime;
}

// printing helper function
void printDiscreteTracks(DiscreteTracks dts)
{
    printDiscreteTracks(dts.tracks, dts.numTracks, dts.trackLength);
} 


void printDiscreteTracks(SoundEvent **tracks, const int numTracks, const int trackLength) 
{
    for (int curTrack = 0; curTrack < numTracks; curTrack++) {
        cout << "Track: " << curTrack << endl;
        for (int curEvent = 0; curEvent < trackLength; curEvent++) {
            cout << "Index: " << curEvent << " : " ;
            printSoundEvent(tracks[curTrack][curEvent]);
        }
    }
}

void printSoundEvent(SoundEvent s)
{
    cout << "Volume: " << s.volume << " Pitch: " << s.pitch << endl;
}


void freeDiscreteTracks(DiscreteTracks dts)
{
    for (int track = 0; track < dts.numTracks; track++)
    {
        free(dts.tracks[track]);
    }
    free(dts.tracks);
}