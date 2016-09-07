
#CUDA_PATH       ?= /usr/local/cuda-7.5

FLAGS := 
FLAGS += -L/usr/local/cuda/lib64/stubs
FLAGS += -L/usr/local/cuda/lib64/
FLAGS += -I/usr/local/cuda-7.5/targets/x86_64-linux/include
#FLAGS += -Icommon/inc

LIBS= -lcuda -lcudart -lcufft -lglut -lGL

all: ex3

ex3: interface.o gpu_main.o animate.o myAIFF.o
	g++  -o ex3 interface.o gpu_main.o animate.o myAIFF.o $(FLAGS) $(LIBS)

interface.o: interface.cpp interface.h params.h animate.h animate.cu
	g++ -w -c interface.cpp $(FLAGS)

gpu_main.o: gpu_main.cu gpu_main.h params.h
	nvcc -w -c gpu_main.cu

animate.o: animate.cu animate.h gpu_main.h params.h
	nvcc -w -c animate.cu

myAIFF.o: myAIFF.cpp myAIFF.h interface.cpp
	g++ -w -c myAIFF.cpp

clean:
	rm *.o;


