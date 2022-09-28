// g++ -msse4 -mavx -march=native -O3 -fopt-info-vec-optimized -fopt-info-vec-missed -o problem5 problem5.cpp

#include <cassert>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <emmintrin.h>
#include <immintrin.h>
#include <iostream>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

#define N (1 << 16)
#define SSE_WIDTH 128
#define AVX2_WIDTH 256

void print_array(int* array);

__attribute__((optimize("no-tree-vectorize"))) int ref_version(int* __restrict__ source,
                                                               int* __restrict__ dest) {
  __builtin_assume_aligned(source, 64);
  __builtin_assume_aligned(dest, 64);

  int tmp = 0;
  for (int i = 0; i < N; i++) {
    tmp += source[i];
    dest[i] = tmp;
  }
  return tmp;
}


long long par_sum_omp_red(int * A,int *dst,int start) {
  // SB: Write your OpenMP code here
  
  long long sum = 0;
  #pragma omp parallel num_threads(4)
  {  
    #pragma omp for nowait reduction(+:sum)
    for(int i=(start*N)/8;i<(start+1)*N/8;i++){
        sum += A[i];
        dst[i]=sum;
    }
  }
  return sum;
}

int omp_version(int* source, int* dest) { 
  
  
  long long block[8];
  #pragma omp parallel for
  for(int i=0;i<8;i++){
    block[i]=par_sum_omp_red(source,dest,i);
    // cout<<block[i]<<endl;
  }

  for(int i=1;i<8;i++){
    block[i]+=block[i-1];
    // cout<<block[i]<<endl;
  }


  #pragma omp parallel for
  for(int i=N/8;i<N;i++){
    dest[i]=dest[i]+block[(i*8)/N-1];
  }



  
  return dest[N-1]; }

int sse4_version(int* source, int* dest) { 
  
  __m128i m1,m2,m3;
  m2=_mm_set1_epi32(0);
  for(int i=0;i<N;i+=4){
    m1=_mm_load_si128((__m128i*)(&source[i]));
    m1=_mm_add_epi32(_mm_bslli_si128(m1,0b00000100),m1);
    m1=_mm_add_epi32(_mm_bslli_si128(m1,0b00001000),m1);
    m1=_mm_add_epi32(_mm_bslli_si128(m1,0b00010000),m1);
    m1=_mm_add_epi32(m1,m2);

    _mm_store_si128((__m128i*)&dest[i],m1);
    m2=_mm_shuffle_epi32(m1,0b11111111);
  }
  

  // cout<<dest[N-1]<<endl;

  return dest[N-1]; }

int avx2_version(int* source, int* dest) { 
  
  __m256i m1,m2,m3;
  __m128i m4,m5;
  int a1,a2;
  m3=_mm256_set1_epi32(0);
  m2=_mm256_set1_epi32(0);
  for(int i=0;i<N;i+=8){
    m1=_mm256_load_si256((__m256i*)&source[i]);
    m1=_mm256_add_epi32(_mm256_bslli_epi128(m1,0b00000100),m1);
    m1=_mm256_add_epi32(_mm256_bslli_epi128(m1,0b00001000),m1);
    m1=_mm256_add_epi32(_mm256_bslli_epi128(m1,0b00010000),m1);
    a1=_mm256_extract_epi32(m1,0b011);
    m2=_mm256_set_epi32(a1,a1,a1,a1,0,0,0,0);
    m1=_mm256_add_epi32(m1,m2);
    m1=_mm256_add_epi32(m1,m3);
    _mm256_store_si256((__m256i*)&dest[i],m1);
    a2=_mm256_extract_epi32(m1,0b111);
    m3=_mm256_set1_epi32(a2);
    
  }
  // cout<<dest[N-1]<<endl;
    return dest[N-1];
  
  
  
}

int* array = nullptr;
int* ref_res = nullptr;
int* omp_res = nullptr;
int* sse_res = nullptr;
int* avx2_res = nullptr;

__attribute__((optimize("no-tree-vectorize"))) int main() {
  array = static_cast<int*>(aligned_alloc(64, N * sizeof(int)));
  ref_res = static_cast<int*>(aligned_alloc(64, N * sizeof(int)));
  omp_res = static_cast<int*>(aligned_alloc(64, N * sizeof(int)));
  sse_res = static_cast<int*>(aligned_alloc(64, N * sizeof(int)));
  avx2_res = static_cast<int*>(aligned_alloc(64, N * sizeof(int)));

  for (int i = 0; i < N; i++) {
    array[i] = 1;
    ref_res[i] = 0;
    omp_res[i] = 0;
    sse_res[i] = 0;
    avx2_res[i] = 0;
  }

  HRTimer start = HR::now();
  int val_ser = ref_version(array, ref_res);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial version: " << val_ser << " time: " << duration << endl;

  start = HR::now();
  int val_omp = omp_version(array, omp_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_omp || printf("OMP result is wrong!\n"));
  cout << "OMP version: " << val_omp << " time: " << duration << endl;

  start = HR::now();
  int val_sse = sse4_version(array, sse_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_sse || printf("SSE4 result is wrong!\n"));
  cout << "SSE4 version: " << val_sse << " time: " << duration << endl;

  start = HR::now();
  int val_avx = avx2_version(array, avx2_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_avx || printf("AVX2 result is wrong!\n"));
  cout << "AVX2 version: " << val_avx << " time: " << duration << endl;

  return EXIT_SUCCESS;
}

void print_array(int* array) {
  for (int i = 0; i < N; i++) {
    cout << array[i] << "\t";
  }
  cout << "\n";
}
