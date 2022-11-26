#!/bin/bash

g++ 190477-prob1.cpp --std=c++11 -fopenmp -ltbb  -o fibonacci
./fibonacci

g++ -std=c++11 -fopenmp 190477-prob2.cpp -o quicksort
./quicksort

g++ 190477-prob3.cpp --std=c++11 -fopenmp -ltbb  -o pi
./pi

g++ 190477-prob4.cpp --std=c++11 -fopenmp -ltbb  -o findmax
./findmax
