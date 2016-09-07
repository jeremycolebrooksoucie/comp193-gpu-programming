Jeremy Colebrook-Soucie
5/18/2016
Music Visualizer

USAGE:

	-a 
		flag that sets afterimage 
		default is off

	-b filepath
		sets BMP file to use
		default is all black image

	-m filepath
		sets midi file to visualize
		required, no default

	-s anInteger
		sets number of discrete samples to take from midi per second
		default is 3

	-v 
		flag that sets verbosity
		default is off



Architecture:

interface	: handles flag parsing, run loop
			  also wrapper functiosn around other modules
			  	(a wrapper around bmp module used, a geometry generator, 
			  	 a wrapper around midiparse)

gpu_main	: handles almost all gpu functions including:
				functions gpu memory allocation and copying into 3 struct below
				device functions 
				normal c functions wrapping device functions
			  defines 3 structs , 
			  	GPU_Midi - containing pointer to midi data in gpu and metadata
			  	GPU_Pallete - containing pointers to rgb data in gpu
			  		and meta data
			  	GPU_Geometries - contianing pointer to geometry data in gpu
			  		and meta data


geometry	: .h file that exports in theory extensible interface for 
				representing geometric shapes with various properties in 
				constant space for each shape
			  this in theory allows for heterogenous arrays of different shapes
			  	without nested pointers
			  only circles are implemented/used

midiparse	: uses midifile package to load a midifile
			  then discretizes/samples from this midifile to generate
			  	a discrete array of sounds

params		: .h file exporting struct AParams that contains flags/other 
				global metadata

animate		: handles displaying 



TODOs:

Aside from the TODOs contained in code, the following higher level revisions 
	are still needed

	-standardize naming convention of GPU_* structs / AParams across functions
	-ensure standardization of struct passing. Goal is to avoid pointers
	-move functions that deal with geometries and params from interface.cpp 
		to geometry.c and params.c
	-add output file option to display end image
	-there is little potential for code reuse in geometry update/geometry draw helper 
		functios in gpu_main. This will cause problems if more geometries are added



Building:

Makefile should work on computer in lab 120 in halligan with following exception
	Relies on midifile package from http://midifile.sapp.org/ to be built
	This package should be a level above the Makefile


Reference/Credit:
	http://midifile.sapp.org/ used for midifile parsing
	http://qdbmp.sourceforge.net/ used for bmp parsing
