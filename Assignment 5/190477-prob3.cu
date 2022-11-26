// nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p2.cu -o assignment5-p2

#include <cmath>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

const uint64_t N = (1 << 10);
#define BLOCK_SIZE 32

using std::cerr;
using std::cout;
using std::endl;

void ErrorCheck(int i , cudaError_t status) {
  if (status != cudaSuccess) {
    cout<<i<<endl;
    cerr << cudaGetErrorString(status) << endl;
  }
}

// __global__ void kernel1(const uint64_t* d_A, const uint64_t* d_B, uint64_t* d_C) {
//   // TODO: Fill in
//   int tid_i = blockIdx.y * blockDim.y + threadIdx.y;
//   int tid_j = blockIdx.x * blockDim.x + threadIdx.x;
//   uint64_t sum = 0;
//   for(uint64_t k = 0; k < N; k++) {
//     sum += d_A[tid_i*N+k] * d_B[k*N+tid_j];
//   }
//   d_C[tid_i*N+tid_j] = sum;
// }

__global__ void kernel2(const uint64_t* d_A, const uint64_t* d_B, uint64_t* d_C) {
  int tid_i = blockIdx.y * blockDim.y + threadIdx.y;
  int tid_j = blockIdx.x * blockDim.x + threadIdx.x;
  int dis_i = threadIdx.y;
  int dis_j = threadIdx.x;
  uint64_t sum = 0;
  for(int k=0; k<N/BLOCK_SIZE; k++) {
    __shared__ double a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double b[BLOCK_SIZE][BLOCK_SIZE];
    a[dis_i][dis_j] = d_A[tid_i*N + (k*BLOCK_SIZE +dis_j)];
    b[dis_i][dis_j] = d_B[(k*BLOCK_SIZE + dis_i)*N + tid_j];
    __syncthreads();
    for(int l = 0; l < BLOCK_SIZE; l+=1) {
      sum = sum + a[dis_i][l] * b[l][dis_j];
            // + a[dis_i][l+1] * b[l+1][dis_j]+ a[dis_i][l+2] * b[l+2][dis_j]+ a[dis_i][l+3] * b[l+3][dis_j]
            // + a[dis_i][l+4] * b[l+4][dis_j]+ a[dis_i][l+5] * b[l+5][dis_j] + a[dis_i][l+6] * b[l+6][dis_j]+ a[dis_i][l+7] * b[l+7][dis_j];
    }
  }
  __syncthreads();
  d_C[tid_i*N+tid_j] = sum;  
}

__host__ void cpumatMul(const uint64_t* h_A, const uint64_t* h_B, uint64_t* h_C) {
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      float sum = 0.0;
      for (uint64_t k = 0; k < N; k++) {
        sum += h_A[i * N + k] * h_B[k * N + j];
      }
      h_C[i * N + j] = sum;
    }
  }
}

__host__ void check_result(const uint64_t* w_ref, const uint64_t* w_opt) {
  bool wrong = false;
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      if (w_ref[i * N + j] != w_opt[i * N + j]) {
        wrong = true;
        goto out;
      }
    }
  }
out:
  if (wrong) {
    cout << " Diffs found!" << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main() {
  const uint64_t SIZE = N * N;

  uint64_t *h_A, *h_B, *h_cpu_C, *h_gpu1_C, *h_gpu2_C;

  h_A = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_B = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_cpu_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_gpu1_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_gpu2_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      h_A[i * N + j] = rand() % 64;
      h_B[i * N + j] = 2;
      h_cpu_C[i * N + j] = 0;
      h_gpu1_C[i * N + j] = 0;
      h_gpu2_C[i * N + j] = 0;
    }
  }

  double clkbegin = rtclock();
  cpumatMul(h_A, h_B, h_cpu_C);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Matmul time on CPU: " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;

  uint64_t *d_A, *d_B, *d_C1;
  // ErrorCheck(1,cudaMalloc(&d_A, SIZE * sizeof(uint64_t)));
  // ErrorCheck(2,cudaMalloc(&d_B, SIZE * sizeof(uint64_t)));
  // ErrorCheck(3,cudaMalloc(&d_C1, SIZE * sizeof(uint64_t)));


  // ErrorCheck(1,cudaHostAlloc((void**)&d_A, SIZE * sizeof(uint64_t),cudaHostAllocDefault));
  // ErrorCheck(2,cudaHostAlloc((void**)&d_B, SIZE * sizeof(uint64_t),cudaHostAllocDefault));
  // ErrorCheck(3,cudaHostAlloc((void**)&d_C1, SIZE * sizeof(uint64_t),cudaHostAllocDefault));

  // ErrorCheck(1,cudaHostAlloc((void**)&d_A, SIZE * sizeof(uint64_t),cudaHostAllocMapped));
  // ErrorCheck(2,cudaHostAlloc((void**)&d_B, SIZE * sizeof(uint64_t),cudaHostAllocMapped));
  // ErrorCheck(3,cudaHostAlloc((void**)&d_C1, SIZE * sizeof(uint64_t),cudaHostAllocMapped));

  // ErrorCheck(1,cudaMallocManaged((void**)&d_A, SIZE * sizeof(uint64_t)));
  // ErrorCheck(2,cudaMallocManaged((void**)&d_B, SIZE * sizeof(uint64_t)));
  // ErrorCheck(3,cudaMallocManaged((void**)&d_C1, SIZE * sizeof(uint64_t)));


  ErrorCheck(4,cudaEventCreate(&start));
  ErrorCheck(5,cudaEventCreate(&end));
  ErrorCheck(6,cudaEventRecord(start, 0));
  ErrorCheck(7,cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice));
  ErrorCheck(8,cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice));
  dim3 TPB1(32, 32);
  dim3 NB1((N+TPB1.x-1)/TPB1.x, (N+TPB1.y-1)/TPB1.y);
  kernel2<<<NB1, TPB1>>>(d_A, d_B, d_C1);
  ErrorCheck(9,cudaMemcpy(h_gpu1_C, d_C1, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  ErrorCheck(10,cudaEventRecord(end, 0));
  ErrorCheck(11,cudaEventSynchronize(end));
  check_result(h_cpu_C, h_gpu1_C);
  float kernel_time;
  ErrorCheck(12,cudaEventElapsedTime(&kernel_time, start, end));
  std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";


//####      OPTIMISED KERNEL          ####################

  
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C1);

  free(h_A);
  free(h_B);
  free(h_cpu_C);
  free(h_gpu1_C);
  free(h_gpu2_C);

  return EXIT_SUCCESS;
}