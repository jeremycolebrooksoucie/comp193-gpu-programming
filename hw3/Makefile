
FLAGS=  -L/usr/local/cuda/lib64/stubs  \
		-L/usr/local/cuda/lib64 \
		-I/usr/local/cuda-7.5/targets/x86_64-linux/include \
		-lcuda -lcudart \
		-L../midifile/lib \
		-I../midifile/include

LIBS=	../midifile/lib/libmidifile.a 

ANIMLIBS= -lglut -lGL

all: ex3

ex3: interface.o gpu_main.o animate.o qdbmp.o midiparse.o
	g++ -std=c++11 -o ex3 interface.o gpu_main.o \
		 animate.o qdbmp.o midiparse.o $(FLAGS) $(ANIMLIBS) $(LIBS)

interface.o: interface.cpp interface.h params.h animate.h animate.cu 
	g++ -w -c interface.cpp $(FLAGS)

gpu_main.o: gpu_main.cu gpu_main.h params.h
	nvcc -w -c gpu_main.cu

animate.o: animate.cu animate.h gpu_main.h params.h
	nvcc -w -c animate.cu

qdbmp.o: qdbmp.c qdbmp.h
	g++ -w -c qdbmp.c 

midiparse.o: midiparse.c midiparse.h
	g++ -std=c++11 -w -c midiparse.c   $(FLAGS) $(LIBS)


miditest: midiparse.c midiparse.h
	g++ -std=c++11 -o miditest midiparse.c   $(FLAGS) $(LIBS)

clean:
	rm interface.o;
	rm gpu_main.o;
	rm animate.o;

