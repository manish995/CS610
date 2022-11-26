// Compile: g++ -std=c++11 -fopenmp -ltbb fibonacci.cpp -o fibonacci

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <tbb/tbb.h>
#include <tbb/task.h>

#define N 40
#define Cutoff 30



using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

using namespace tbb;

// Serial Fibonacci
long ser_fib(int n) {
  if (n == 0 || n == 1) {
    return (n);
  }
  return ser_fib(n - 1) + ser_fib(n - 2);
}




long omp_fib_v1(int n) {
  // TODO: Implement OpenMP version with explicit tasks
  if(n==0)return 0;
  if(n==1)return 1;
  long x=0,y=0;
  #pragma omp task shared(x)
  x =omp_fib_v1(n-1);
  #pragma omp task shared(y)
  y=omp_fib_v1(n-2);
  
  #pragma omp taskwait

  return x+y;
}

long omp_fib_v1_wrapper(int n){
  long long ans;
  #pragma omp parallel num_threads(5)
  {
    #pragma omp single
    {
    ans = omp_fib_v1(n);
    }
  }
  return ans;
}






long omp_fib_v2(int n) {
  // TODO: Implement an optimized OpenMP version with any valid optimization
  if(n==0)return 0;
  if(n==1)return 1;
  
  if (n > Cutoff)
  {
    long long x, y;
    #pragma omp task shared(x)
      x = omp_fib_v2(n - 1);
    #pragma omp task shared(y)
      y = omp_fib_v2(n - 2);
    #pragma omp taskwait

    return x+y;
  }
  else
  {
    return ser_fib(n);
  }
}

long omp_fib_v2_wrapper(int n){
  long long ans;
  #pragma omp parallel num_threads(5)
  {
    #pragma omp single
    {
    ans = omp_fib_v2(n);
    }
  }
  return ans;
}





class fibtask: public task{
public:
  const long n;
  long * const fib;
  fibtask(long num, long * fibo) : n(num) ,fib(fibo) {}

  task* execute(){
    if(n>Cutoff){
      long fib1,fib2;
      fibtask &task1 = *new(allocate_child()) fibtask(n-1,&fib1);
      fibtask &task2 = *new(allocate_child()) fibtask(n-2,&fib2);

      set_ref_count(3);
      spawn(task2);
      spawn_and_wait_for_all(task1);

      *fib = fib1 + fib2;
    }
    else
    {
      *fib = ser_fib(n);
    }
  return NULL;
  }
};


long tbb_fib_blocking(int n) {
  // TODO: Implement Intel TBB version with blocking style
  long fib;
  fibtask &parent = *new(task::allocate_root()) fibtask(n, &fib);
  task::spawn_root_and_wait(parent);
  return fib;

}







class fibc: public task {
public:
  long* const fib;
  long fib1, fib2;
  fibc(long *fibo): fib(fibo) {}

  task* execute() {
    *fib = fib1 + fib2;
    return NULL;
  }
};

class fibtaskc: public task {
public:
  const int n;
  long* const fib;
  fibtaskc(int n_, long* fib_): n(n_), fib(fib_) {}

  task* execute() {
    if(n > Cutoff)
    {
      fibc &parent = *new(allocate_continuation())fibc(fib);
      fibtaskc &task1 = *new(parent.allocate_child())fibtaskc(n-1, &parent.fib1);
      fibtaskc &task2 = *new(parent.allocate_child())fibtaskc(n-2, &parent.fib2);
      parent.set_ref_count(2);
      spawn(task2);
      spawn(task1);
    }
    else 
    {
      *fib = ser_fib(n);
    } 
    return NULL;
  }
};

long tbb_fib_cps(int n) {
  long fib = 0;
  fibtaskc &task = *new(task::allocate_root())fibtaskc(n, &fib);
  task::spawn_root_and_wait(task);
  return fib;
}






int main(int argc, char** argv) {
  cout << std::fixed << std::setprecision(5);

  HRTimer start = HR::now();
  long s_fib = ser_fib(N);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial time: " << duration << " microseconds" << endl;

  start = HR::now();
  long omp_v1 = omp_fib_v1_wrapper(N);
  end = HR::now();
  assert(s_fib == omp_v1);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "OMP v1 time: " << duration << " microseconds" << endl;

  start = HR::now();
  long omp_v2;
  omp_v2 = omp_fib_v2_wrapper(N);
  end = HR::now();
  assert(s_fib == omp_v2);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "OMP v2 time: " << duration << " microseconds" << endl;

  start = HR::now();
  long blocking_fib = tbb_fib_blocking(N);
  end = HR::now();
  assert(s_fib == blocking_fib);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "TBB (blocking) time: " << duration << " microseconds" << endl;

  start = HR::now();
  long cps_fib = tbb_fib_cps(N);
  end = HR::now();
  assert(s_fib == cps_fib);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "TBB (cps) time: " << duration << " microseconds" << endl;

  return EXIT_SUCCESS;
}
