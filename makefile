NVCC = nvcc

all: predict.exe

predict.exe: main.o predict.o
	nvcc -arch=sm_20 main.o predict.o -o predict.exe -std=c++11

predict.o: predict.cu
	nvcc -arch=sm_20 -c predict.cu -o predict.o -std=c++11

main.o: main.cc
	g++ -c main.cc -o main.o -std=c++11

clean:
	rm *.o *.exe
