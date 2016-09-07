
FLAGS=  -L/usr/local/cuda/lib64/stubs  -L/usr/local/cuda/lib64 -I/usr/local/cuda-7.5/targets/x86_64-linux/include -lcuda -lcudart
ANIMLIBS= -lglut -lGL

all: ex2

ex2: interface.o gpu_main.o animate.o qdbmp.o
	g++ -o ex2 interface.o gpu_main.o animate.o qdbmp.o $(FLAGS) $(ANIMLIBS)

interface.o: interface.cpp interface.h params.h animate.h animate.cu 
	g++ -w -c interface.cpp $(FLAGS)

gpu_main.o: gpu_main.cu gpu_main.h params.h
	nvcc -w -c gpu_main.cu

animate.o: animate.cu animate.h gpu_main.h params.h
	nvcc -w -c animate.cu

qdbmp.o: qdbmp.c qdbmp.h
	g++ -w -c qdbmp.c 

clean:
	rm interface.o;
	rm gpu_main.o;
	rm animate.o;

