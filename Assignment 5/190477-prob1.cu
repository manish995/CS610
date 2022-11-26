// nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p1.cu -o assignment5-p1

#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <iostream>
#include <new>
#include <sys/time.h>

#define THRESHOLD (0.000001)

#define SIZE1 8192
#define SIZE2 8200
#define ITER 100

using std::cerr;
using std::cout;
using std::endl;


//#      KERNEL 1D    ####

__global__ void kernel1(const double *d_k1_in, double *d_k1_out) {
  // TODO: Fill in
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < SIZE1 - 1) {
    for(int k = 0; k < ITER; k++) {
      for(int i = 1; i < (SIZE1 - 1); i++) {
        d_k1_out[i*SIZE1 + j + 1] =
              (d_k1_in[(i - 1)*SIZE1 + j + 1] + d_k1_in[i*SIZE1 + j + 1] + d_k1_in[(i + 1)*SIZE1 +j + 1]);
      }
    }
  }

}


//#          KERNEL 2D               ######

__global__ void kernel2(double *d_k2_in, double *d_k2_out) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y +threadIdx.y;
    if((j>=0) && (j < SIZE2-1 )&& (i>=1 )&& (i<SIZE2-1)){
        for(int k=0;k<ITER;k++){
            d_k2_out[i*SIZE2 + j + 1] =
              (d_k2_in[(i - 1)*SIZE2 + j + 1] + d_k2_in[i*SIZE2 + j + 1] + d_k2_in[(i + 1)*SIZE2 +j + 1]);
        }
    }

}
//#        KERNEL 3D                 #######
__global__ void kernel3(double *d_k2_in, double *d_k2_out) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y +threadIdx.y;
    int k = blockIdx.z *blockDim.z + threadIdx.z;

    if((j>=0) && (j < SIZE2-1 )&& (i>=1 )&& (i<SIZE2-1) && (k<ITER)){
            d_k2_out[i*SIZE2 + j + 1] =
              (d_k2_in[(i - 1)*SIZE2 + j + 1] + d_k2_in[i*SIZE2 + j + 1] + d_k2_in[(i + 1)*SIZE2 +j + 1]);
    }

}


__host__ void serial(const double *h_ser_in, double *h_ser_out) {
  for (int k = 0; k < ITER; k++) {
    for (int i = 1; i < (SIZE1 - 1); i++) {
      for (int j = 0; j < (SIZE1 - 1); j++) {
        h_ser_out[i * SIZE1 + j + 1] =
            (h_ser_in[(i - 1) * SIZE1 + j + 1] + h_ser_in[i * SIZE1 + j + 1] +
             h_ser_in[(i + 1) * SIZE1 + j + 1]);
      }
    }
  }
}

__host__ void serial2(const double *h_ser_in, double *h_ser_out) {
  for (int k = 0; k < ITER; k++) {
    for (int i = 1; i < (SIZE2 - 1); i++) {
      for (int j = 0; j < (SIZE2 - 1); j++) {
        h_ser_out[i * SIZE2 + j + 1] =
            (h_ser_in[(i - 1) * SIZE2 + j + 1] + h_ser_in[i * SIZE2 + j + 1] +
             h_ser_in[(i + 1) * SIZE2 + j + 1]);
      }
    }
  }
}





__host__ void check_result(const double *w_ref, const double *w_opt,
                           const uint64_t size) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      double this_diff = w_ref[i * size + j] - w_opt[i * size + j];
      if (fabs(this_diff) > THRESHOLD) {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

void ErrorCheck(int i, cudaError_t status) {
  if (status != cudaSuccess)
  { 
    cout<<i<<endl;
    cerr << cudaGetErrorString(status) << endl;
  }
}

__host__ double rtclock() { // Seconds
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
  double *h_ser_in = new double[SIZE1 * SIZE1];
  double *h_ser_out = new double[SIZE1 * SIZE1];

  double *h_k1_in = new double[SIZE1 * SIZE1];
  double *h_k1_out = new double[SIZE1 * SIZE1];

  for (int i = 0; i < SIZE1; i++) {
    for (int j = 0; j < SIZE1; j++) {
      h_ser_in[i * SIZE1 + j] = 1;
      h_ser_out[i * SIZE1 + j] = 1;
      h_k1_in[i * SIZE1 + j] = 1;
      h_k1_out[i * SIZE1 + j] = 1;
    }
  }

  double *h_k2_in = new double[SIZE2 * SIZE2];
  double *h_k2_out = new double[SIZE2 * SIZE2];
  double *h_ser2_in = new double[SIZE2 * SIZE2];
  double *h_ser2_out = new double[SIZE2 * SIZE2];

  for (int i = 0; i < SIZE2; i++) {
    for (int j = 0; j < SIZE2; j++) {     
      h_ser2_in[i * SIZE2 + j] = 1;
      h_ser2_out[i * SIZE2 + j] = 0;
      h_k2_in[i * SIZE2 + j] = 1;
      h_k2_out[i * SIZE2 + j] = 0;
    }
  }



  //#### FIRST KERNEL TESTING    #####

   double clkbegin = rtclock();
   serial(h_ser_in, h_ser_out);
   double clkend = rtclock();
   double time = clkend - clkbegin; // seconds
   cout << "Serial code on CPU: " << ((2.0 * SIZE1 * SIZE1 * ITER) / time)
        << " GFLOPS; Time = " << time * 1000 << " msec" << endl;


   cudaError_t status;
   cudaEvent_t start, end;
  float k1_time; // ms
  double *d_k1_in;
  double *d_k1_out;
  ErrorCheck(1,cudaEventCreate(&start));
  ErrorCheck(2,cudaEventCreate(&end));
  ErrorCheck(3,cudaMalloc(&d_k1_in, SIZE1*SIZE1*sizeof(double)));
  ErrorCheck(4,cudaMalloc(&d_k1_out, SIZE1*SIZE1*sizeof(double)));
  ErrorCheck(5,cudaEventRecord(start,0));
  ErrorCheck(6,cudaMemcpy(d_k1_in, h_k1_in, SIZE1*SIZE1*sizeof(double), cudaMemcpyHostToDevice));
  ErrorCheck(7,cudaMemcpy(d_k1_out, h_k1_out, SIZE1*SIZE1*sizeof(double), cudaMemcpyHostToDevice));
  dim3 theadsPerblock(1024);
  dim3 numBlocks(SIZE1 / theadsPerblock.x);
  kernel1<<<numBlocks, theadsPerblock>>>(d_k1_in,d_k1_out);
  ErrorCheck(8,cudaMemcpy(h_k1_out, d_k1_out, SIZE1*SIZE1*sizeof(double), cudaMemcpyDeviceToHost));
  ErrorCheck(9,cudaEventRecord(end,0));
  ErrorCheck(10,cudaEventSynchronize(end));
  ErrorCheck(11,cudaEventElapsedTime(&k1_time,start,end));
  check_result(h_ser_out, h_k1_out, SIZE1);
  cout << "Kernel 1 on GPU: "
       << ((2.0 * SIZE1 * SIZE1 * ITER) / (k1_time * 1.0e-3))
       << " GFLOPS; Time = " << k1_time << " msec" << endl;





  // ###########          SECOND KERNEL TESTING                  #################
 
  double clkbegin2 = rtclock();
  serial2(h_ser2_in, h_ser2_out);
  double clkend2 = rtclock();
  double time2 = clkend2 - clkbegin2; // seconds
  cout << "Serial2 code on CPU: " << ((2.0 * SIZE2 * SIZE2 * ITER) / time2)
       << " GFLOPS; Time = " << time2 * 1000 << " msec" << endl;

  double *d_k2_in;
  double *d_k2_out;
  float k2_time;
  ErrorCheck(12,cudaEventCreate(&start));
  ErrorCheck(13,cudaEventCreate(&end));
  ErrorCheck(14,cudaMalloc(&d_k2_in, SIZE2*SIZE2*sizeof(double)));
  ErrorCheck(15,cudaMalloc(&d_k2_out, SIZE2*SIZE2*sizeof(double)));
  ErrorCheck(16,cudaEventRecord(start,0));
  ErrorCheck(17,cudaMemcpy(d_k2_in, h_k2_in, SIZE2*SIZE2*sizeof(double), cudaMemcpyHostToDevice));
  ErrorCheck(18,cudaMemcpy(d_k2_out, h_k2_out, SIZE2*SIZE2*sizeof(double), cudaMemcpyHostToDevice));
  dim3 theadsPerblock2(32,16);
  dim3 numBlocks2((SIZE2 + theadsPerblock2.x - 1) / theadsPerblock2.x,(SIZE2 + theadsPerblock2.y - 1) / theadsPerblock2.y);
  kernel2<<<numBlocks2, theadsPerblock2>>>(d_k2_in,d_k2_out);
  ErrorCheck(19,cudaMemcpy(h_k2_out, d_k2_out, SIZE2*SIZE2*sizeof(double), cudaMemcpyDeviceToHost));
  ErrorCheck(20,cudaEventRecord(end,0));
  ErrorCheck(21,cudaEventSynchronize(end));
  ErrorCheck(22,cudaEventElapsedTime(&k2_time,start,end));
  ErrorCheck(23,cudaEventDestroy(start));
  ErrorCheck(24,cudaEventDestroy(end));
  check_result(h_ser2_out, h_k2_out, SIZE2);
  cout << "Kernel 2 on GPU: "
       << ((2.0 * SIZE2 * SIZE2 * ITER) / (k2_time * 1.0e-3))
       << " GFLOPS; Time = " << k2_time << " msec" << endl;

  
  // ###########          THIRD KERNEL TESTING                  #################

  double *d_k3_in;
  double *d_k3_out;
  float k3_time;
  ErrorCheck(25,cudaEventCreate(&start));
  ErrorCheck(26,cudaEventCreate(&end));
  ErrorCheck(27,cudaMalloc(&d_k3_in, SIZE2*SIZE2*sizeof(double)));
  ErrorCheck(28,cudaMalloc(&d_k3_out, SIZE2*SIZE2*sizeof(double)));
  ErrorCheck(29,cudaEventRecord(start,0));
  ErrorCheck(30,cudaMemcpy(d_k3_in, h_k2_in, SIZE2*SIZE2*sizeof(double), cudaMemcpyHostToDevice));
  ErrorCheck(31,cudaMemcpy(d_k3_out, h_k2_out, SIZE2*SIZE2*sizeof(double), cudaMemcpyHostToDevice));
  dim3 theadsPerblock3(16,16,4);
  dim3 numBlocks3((SIZE2 + theadsPerblock3.x - 1) / theadsPerblock3.x, (SIZE2 + theadsPerblock3.y - 1) / theadsPerblock3.y , ITER/theadsPerblock3.z);
  kernel3<<<numBlocks3, theadsPerblock3>>>(d_k3_in,d_k3_out);
  ErrorCheck(32,cudaMemcpy(h_k2_out, d_k3_out, SIZE2*SIZE2*sizeof(double), cudaMemcpyDeviceToHost));
  ErrorCheck(33,cudaEventRecord(end,0));
  ErrorCheck(34,cudaEventSynchronize(end));
  ErrorCheck(35,cudaEventElapsedTime(&k3_time,start,end));
  ErrorCheck(36,cudaEventDestroy(start));
  ErrorCheck(37,cudaEventDestroy(end));
  check_result(h_ser2_out, h_k2_out, SIZE2);
  cout << "Kernel 3 on GPU: "
       << ((2.0 * SIZE2 * SIZE2 * ITER) / (k3_time * 1.0e-3))
       << " GFLOPS; Time = " << k3_time << " msec" << endl;



  //####          FOURTH KERNEL TESTING                #########


  // double *d_k4_in;
  // double *d_k4_out;
  // float k4_time;
  // ErrorCheck(25,cudaEventCreate(&start));
  // ErrorCheck(26,cudaEventCreate(&end));
  // ErrorCheck(27,cudaMalloc(&d_k4_in, SIZE2*SIZE2*sizeof(double)));
  // ErrorCheck(28,cudaMalloc(&d_k4_out, SIZE2*SIZE2*sizeof(double)));
  // ErrorCheck(29,cudaEventRecord(start,0));
  // ErrorCheck(30,cudaMemcpy(d_k4_in, h_k2_in, SIZE2*SIZE2*sizeof(double), cudaMemcpyHostToDevice));
  // ErrorCheck(31,cudaMemcpy(d_k4_out, h_k2_out, SIZE2*SIZE2*sizeof(double), cudaMemcpyHostToDevice));
  // dim3 theadsPerblock4(32,32);
  // dim3 numBlocks4((SIZE2 + theadsPerblock4.x - 1) / (theadsPerblock4.x),(SIZE2/2+ theadsPerblock4.y - 1) / (theadsPerblock4.y));
  // // cout<<"threads: "<<numBlocks4<<endl;
  // kernel4<<<numBlocks4, theadsPerblock4>>>(d_k4_in,d_k4_out);
  // ErrorCheck(32,cudaMemcpy(h_k2_out, d_k4_out, SIZE2*SIZE2*sizeof(double), cudaMemcpyDeviceToHost));
  // ErrorCheck(33,cudaEventRecord(end,0));
  // ErrorCheck(34,cudaEventSynchronize(end));
  // ErrorCheck(35,cudaEventElapsedTime(&k4_time,start,end));
  // ErrorCheck(36,cudaEventDestroy(start));
  // ErrorCheck(37,cudaEventDestroy(end));
  // check_result(h_ser2_out, h_k2_out, SIZE2);
  // cout << "Kernel 4 on GPU: "
  //      << ((2.0 * SIZE2 * SIZE2 * ITER) / (k4_time * 1.0e-3))
  //      << " GFLOPS; Time = " << k4_time << " msec" << endl;

  // cudaFree(d_k1_in);
  // cudaFree(d_k1_out);
  // cudaFree(d_k2_in);
  // cudaFree(d_k2_out);
  // cudaFree(d_k3_in);
  // cudaFree(d_k3_out);
  // cudaFree(d_k4_in);
  // cudaFree(d_k4_out);

  delete[] h_ser_in;
  delete[] h_ser_out;
  delete[] h_k1_in;
  delete[] h_k1_out;

  delete[] h_k2_in;
  delete[] h_k2_out;

  return EXIT_SUCCESS;
}

