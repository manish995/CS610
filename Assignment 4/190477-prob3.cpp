// g++ -std=c++11 -fopenmp -ltbb pi.cpp -o pi

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <tbb/tbb.h>
#include <tbb/task.h>
using namespace tbb;

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

const int NUM_INTERVALS = std::numeric_limits<int>::max();

double serial_pi() {
  double dx = 1.0 / NUM_INTERVALS;
  double sum = 0.0;
  for (int i = 0; i < NUM_INTERVALS; ++i) {
    double x = (i + 0.5) * dx;
    double h = std::sqrt(1 - x * x);
    sum += h * dx;
  }
  double pi = 4 * sum;
  return pi;
}

double omp_pi() {
  // TODO: Implement OpenMP version with minimal false sharing

  double dx = 1.0 / NUM_INTERVALS;
  double sum = 0.0;
  
  #pragma omp parallel for reduction(+:sum) num_threads(5)
    for (int i = 0; i < NUM_INTERVALS; i++) {
      double x = (i + 0.5) * dx;
      double h = std::sqrt(1 - x * x);
      sum += h * dx;
    }
  
  double pi = 4*sum;
  return pi;

}

class pi { 
public:
  double dx = 1.0 / NUM_INTERVALS;
  double sum;
  pi(): sum(0.0) {}
  pi(pi &x, split): sum(0.0) {}

  void operator()(const blocked_range<int>& r) {
    double temp = sum;
    for(int i = r.begin(); i != r.end(); i++) {
      double x = (i + 0.5) * dx;
      double h = std::sqrt(1 - x * x);
      temp += h * dx;
    }
    sum = temp;
  }

  void join(const pi &y) {
    sum += y.sum;
  }
};

double tbb_pi() {
  pi res;
  parallel_reduce(blocked_range<int>(0, NUM_INTERVALS), res);
  return (4*res.sum);
}

int main() {
  cout << std::fixed << std::setprecision(5);

  HRTimer start = HR::now();
  double ser_pi = serial_pi();
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial pi: " << ser_pi << " Serial time: " << duration << " microseconds" << endl;

  start = HR::now();
  double o_pi = omp_pi();
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel pi (OMP): " << o_pi << " Parallel time: " << duration << " microseconds"
       << endl;

  start = HR::now();
  double t_pi = tbb_pi();
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel pi (TBB): " << t_pi << " Parallel time: " << duration << " microseconds"
       << endl;

  return EXIT_SUCCESS;
}

