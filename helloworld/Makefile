
FLAGS=  -L/usr/local/cuda/lib64/stubs  -L/usr/local/cuda/lib64 -I/usr/local/cuda-7.5/targets/x86_64-linux/include -lcuda -lcudart


all: helloworld

helloworld: helloworld.cu
	nvcc -w -c helloworld.cu



#ex1: interface.o gpu_main.o
#	g++ -o ex1 interface.o gpu_main.o $(FLAGS)

#interface.o: interface.cpp interface.h
#	g++ -w -c interface.cpp $(FLAGS)
#
#gpu_main.o: gpu_main.cu gpu_main.h
#	nvcc -w -c gpu_main.cu

#clean:
#	rm interface.o;
#	rm gpu_main.o;

