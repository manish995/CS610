// Compile: g++ -std=c++11 -ltbb find-max.cpp -o find-max

#include <cassert>
#include <chrono>
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

#define N (1 << 20)

uint64_t serial_find_max(const uint64_t* a) {
  uint64_t value_of_max = std::numeric_limits<uint64_t>::min();
  uint64_t index_of_max = 0;
  for (uint64_t i = 0; i < N; i++) {
    uint64_t value = a[i];
    if (value > value_of_max) {
      value_of_max = value;
      index_of_max = i;
    }
  }
  return index_of_max;
}

class findmax {
  const uint64_t* arr;
public:
  uint64_t value_of_max;
  uint64_t index_of_max;

  void operator()(const blocked_range<uint64_t>& r) {

    for(uint64_t i = r.begin(); i != r.end(); i++) {
      if (arr[i] > value_of_max) {
        value_of_max = arr[i];
         index_of_max  = i;
      }
    }
  }

  findmax(const uint64_t* Arr): arr(Arr), value_of_max(0), index_of_max(0) {}
  findmax(findmax &x, split): arr(x.arr), value_of_max(0), index_of_max(0) {}

  void join(const findmax &y) {
    if(value_of_max < y.value_of_max) {
      value_of_max = y.value_of_max;
      index_of_max = y.index_of_max;
    }
    else if((value_of_max == y.value_of_max) && (index_of_max > y.index_of_max)){
      value_of_max = y.value_of_max;
      index_of_max = y.index_of_max;
    }
  }
};

uint64_t tbb_find_max(const uint64_t* a) {
  findmax res(a);
  parallel_reduce(blocked_range<uint64_t>(0,N), res);
  return (res.index_of_max);
}

int main() {
  uint64_t* a = new uint64_t[N];
  for (uint64_t i = 0; i < N; i++) {
    a[i] = rand();
  }

  HRTimer start = HR::now();
  uint64_t s_max_idx = serial_find_max(a);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Sequential max index: " << s_max_idx << " in " << duration << " us" << endl;

  start = HR::now();
  uint64_t tbb_max_idx = tbb_find_max(a);
  end = HR::now();
  // cout<<tbb_max_idx<<endl;
  assert(s_max_idx == tbb_max_idx);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel (TBB) max index in " << duration << " us" << endl;

  return EXIT_SUCCESS;
}