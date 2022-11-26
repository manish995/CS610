#I run problem 1 ,2, 4 on gpu3 and problem 3 on gpu2 
nvcc -g -G -arch=sm_61 -std=c++11 190477-prob1.cu -o binary
./binary


nvcc -g -G -arch=sm_61 -std=c++11 190477-prob2.cu -o binary
./binary



nvcc -g -G -arch=sm_61 -std=c++11 190477-prob3.cu -o binary
./binary


nvcc -g -G -arch=sm_61 -std=c++11 190477-prob4.cu -o binary
nvprof ./binary
