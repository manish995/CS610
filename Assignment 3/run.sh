# Problem 2
echo "-------------O2 Level Optimisation------------------"
g++ -O2 -mavx -march=native -o problem2 190477-prob2.cpp
./problem2
echo "-------------O3 Level Optimisation------------------"
g++ -O3 -mavx -march=native -o problem2 190477-prob2.cpp
./problem2

# Problem 3
g++ -mavx -O3 -fopenmp  -o problem3 190477-prob3.cpp
./problem3

# Problem 4

g++ -O2 -fopenmp -o problem4 190477-prob4.cpp
./problem4

# Problem 5

g++ -msse4.1 -mavx -march=native -O3 -fopt-info-vec-optimized -fopt-info-vec-missed -fopenmp -o problem5 190477-prob5.cpp
./problem5