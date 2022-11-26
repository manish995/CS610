// nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p4.cu -o assignment5-p4

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

const uint64_t N = (64);
#define THRESHOLD (0.000001)
#define BLOCK_SIZE 8

using std::cerr;
using std::cout;
using std::endl;

void ErrorCheck(int i , cudaError_t status) {
  if (status != cudaSuccess) {
    cout<<i<<endl;
    cerr << cudaGetErrorString(status) << endl;
  }
}

// TODO: Edit the function definition as required
__global__ void kernel1(double *in,double* out) {

  int i= blockIdx.z * blockDim.z + threadIdx.z;
  int j= blockIdx.y * blockDim.y + threadIdx.y;
  int k= blockIdx.x * blockDim.x + threadIdx.x;
  if((i > 0)&&(i < N-1)&&(j > 0)&&(j < N-1 )&&(k > 0 )&&(k < N-1)) {
    out[i*N*N+j*N+k] = 0.8 * (in[(i-1)*N*N + j*N + k] + in[(i+1)*N*N + j*N + k] + in[i*N*N + (j-1)*N + k] 
                             + in[i*N*N + (j+1)*N + k] + in[i*N*N + j*N + k-1] + in[i*N*N + j*N + k+1]);
  }
}

// TODO: Edit the function definition as required
__global__ void kernel2(double *in,double* out){

  int tid_i = (blockIdx.z * BLOCK_SIZE + threadIdx.z);
  int tid_j = (blockIdx.y * BLOCK_SIZE + threadIdx.y);
  int tid_k = (blockIdx.x * BLOCK_SIZE + threadIdx.x);
  
  int i = threadIdx.z;
  int j = threadIdx.y;
  int k = threadIdx.x;

  __shared__ float a[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];
  a[i][j][k] = in[tid_i*N*N + tid_j*N + tid_k];
  __syncthreads();

  if ((k>0) && (k<BLOCK_SIZE-1) && (j>0) && (j<BLOCK_SIZE-1) && (i>0) && (i<BLOCK_SIZE-1)) {
    out[tid_i*N*N + tid_j*N + tid_k] = 0.8 * (a[i-1][j][k]+a[i+1][j][k]+a[i][j-1][k]+a[i][j+1][k]+a[i][j][k-1]+a[i][j][k+1]);
  } 
  else if((tid_i>0) && (tid_i<N-1) && (tid_j>0) && (tid_j<N-1)&& (tid_k>0) && (tid_k<N-1)) {
      out[tid_i*N*N + tid_j*N + tid_k] = 0.8 * (in[(tid_i-1)*N*N + tid_j*N + tid_k] + in[(tid_i+1)*N*N + tid_j*N + tid_k] + in[tid_i*N*N + (tid_j-1)*N + tid_k] 
                                  + in[tid_i*N*N + (tid_j+1)*N + tid_k] + in[tid_i*N*N + tid_j*N + tid_k-1] + in[tid_i*N*N + tid_j*N + tid_k+1]);
  }
  
}

// TODO: Edit the function definition as required
__host__ void stencil(double *in, double * out) {
  for(int i = 1; i < N - 1; i++) {
    for(int j = 1; j < N - 1; j++) {
      for(int k = 1; k < N - 1; k++) {
        out[i*N*N+j*N+k] = 0.8*(in[(i-1)*N*N + j*N + k] + in[(i+1)*N*N + j*N + k] + in[i*N*N + (j-1)*N + k] 
                             + in[i*N*N + (j+1)*N + k] + in[i*N*N + j*N + k-1] + in[i*N*N + j*N + k+1]);
      }
    }
  }
}

__host__ void check_result(const double* w_ref, const double* w_opt, const uint64_t size) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        this_diff = w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

void print_mat(double* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        printf("%lf,", A[i * N * N + j * N + k]);
      }
      printf("      ");
    }
    printf("\n");
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
  uint64_t SIZE = N * N * N;
  double* h_in = (double*)malloc(SIZE * sizeof(double));
  double* h_out = (double*)malloc(SIZE * sizeof(double));
  double* h_k1_out = (double*)malloc(SIZE * sizeof(double));
  double* h_k2_out = (double*)malloc(SIZE * sizeof(double));

  for(long long i = 0; i < SIZE; ++i) {
    h_in[i] = rand() % 100;
    h_out[i] = 0.0;
    h_k1_out[i] = 0.0;
    h_k2_out[i] = 0.0;
  }
  double clkbegin = rtclock();
  stencil(h_in,h_out);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;
  // TODO: Fill in kernel1
  // TODO: Adapt check_result() and invoke
  double * d_k1_in,*d_k1_out;
  ErrorCheck(1,cudaMalloc(&d_k1_in, SIZE * sizeof(double)));
  ErrorCheck(2,cudaMalloc(&d_k1_out, SIZE * sizeof(double)));
  ErrorCheck(3,cudaEventCreate(&start));
  ErrorCheck(4,cudaEventCreate(&end));
  ErrorCheck(5,cudaEventRecord(start, 0));
  ErrorCheck(6,cudaMemcpy(d_k1_in, h_in, SIZE * sizeof(double), cudaMemcpyHostToDevice));
  ErrorCheck(7,cudaMemcpy(d_k1_out, h_k1_out, SIZE * sizeof(double), cudaMemcpyHostToDevice));
  dim3 TPB1(8, 8, 8);
  dim3 NB1((N+TPB1.x-1)/TPB1.x, (N+TPB1.y-1)/TPB1.y, (N+TPB1.z-1) / TPB1.z);
  kernel1<<<NB1, TPB1>>>(d_k1_in, d_k1_out);
  ErrorCheck(8,cudaMemcpy(h_k1_out, d_k1_out, SIZE * sizeof(double), cudaMemcpyDeviceToHost));
  ErrorCheck(9,cudaEventRecord(end, 0));
  ErrorCheck(10,cudaEventSynchronize(end));
  check_result(h_out, h_k1_out, N);
   float kernel_time;
  ErrorCheck(11,cudaEventElapsedTime(&kernel_time, start, end));
  std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";
  ErrorCheck(12,cudaEventDestroy(start));
  ErrorCheck(13,cudaEventDestroy(end));
  // TODO: Fill in kernel2
  // TODO: Adapt check_result() and invoke






  double *d_k2_in, *d_k2_out;
  // ErrorCheck(1,cudaMalloc(&d_k2_in, SIZE * sizeof(double))); //OUT OF MEMORY ERROR
  ErrorCheck(3,cudaMalloc(&d_k2_out, SIZE * sizeof(double)));
  ErrorCheck(14,cudaEventCreate(&start));
  ErrorCheck(15,cudaEventCreate(&end));
  ErrorCheck(16,cudaEventRecord(start, 0));
  // ErrorCheck(116,cudaMemcpy(d_k2_in, h_in, SIZE * sizeof(double), cudaMemcpyHostToDevice));
  ErrorCheck(17,cudaMemcpy(d_k2_out, h_k2_out, SIZE * sizeof(double), cudaMemcpyHostToDevice));
  dim3 threadsPerBlock2(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks2((N + threadsPerBlock2.x - 1) / threadsPerBlock2.x, (N + threadsPerBlock2.y - 1) / threadsPerBlock2.y, (N + threadsPerBlock2.z - 1) / threadsPerBlock2.z);
  kernel2<<<numBlocks2, threadsPerBlock2>>>(d_k1_in, d_k2_out);
  ErrorCheck(18,cudaMemcpy(h_k2_out, d_k2_out, SIZE * sizeof(double), cudaMemcpyDeviceToHost));
  ErrorCheck(19,cudaEventRecord(end, 0));
  ErrorCheck(20,cudaEventSynchronize(end));
  check_result(h_out, h_k2_out, N);
  ErrorCheck(21,cudaEventElapsedTime(&kernel_time, start, end));
  std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";
  ErrorCheck(22,cudaEventDestroy(start));
  ErrorCheck(23,cudaEventDestroy(end));

  cudaFree(d_k1_in);
  // cudaFree(d_k2_in);
  cudaFree(d_k1_out);
  cudaFree(d_k2_out);

  free(h_in);
  free(h_out);
  free(h_k1_out);
  free(h_k2_out);


  return EXIT_SUCCESS;
}