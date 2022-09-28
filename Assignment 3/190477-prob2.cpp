// Compile: g++ -O2 -mavx -march=native -o problem2 problem2.cpp
// Execute: ./problem2

#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <immintrin.h>


using std::cout;
using std::endl;
using std::ios;

const int N = (1 << 13);
const int Niter = 10;
const double THRESHOLD = 0.000001;

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << endl;
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void reference(double** A, double* x, double* y_ref, double* z_ref) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      y_ref[j] = y_ref[j] + A[i][j] * x[i];
      z_ref[j] = z_ref[j] + A[j][i] * x[i];
    }
  }
}

void check_result(double* w_ref, double* w_opt) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++) {
    this_diff = w_ref[i] - w_opt[i];
    if (fabs(this_diff) > THRESHOLD) {
      // cout<<""
      numdiffs++;
      if (this_diff > maxdiff)
        maxdiff = this_diff;
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over threshold " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

// TODO: INITIALLY IDENTICAL TO REFERENCE; MAKE YOUR CHANGES TO OPTIMIZE THE CODE
// You can create multiple versions of the optimized() function to test your changes

void optimized1(double** __restrict__ A, double* __restrict__ x, double* __restrict__ y_opt, double* __restrict__ z_opt) {
  
  #pragma GCC ivdep
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      z_opt[j] = z_opt[j] + A[j][i] * x[i];
    }
  }
  
}


void optimized2(double** __restrict__ A, double* __restrict__ x, double* __restrict__ y_opt, double* __restrict__ z_opt) {
  
  
  #pragma GCC ivdep
  
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      z_opt[j] = z_opt[j] + A[j][i] * x[i];
    }
  }
}


void optimized8(double** __restrict__ A, double* __restrict__ x, double* __restrict__ y_opt, double* __restrict__ z_opt) {
  
  
  #pragma GCC ivdep
  
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j=j+4) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      z_opt[j] = z_opt[j] + A[j][i] * x[i];
      y_opt[j+1] = y_opt[j+1] + A[i][j+1] * x[i];
      z_opt[j+1] = z_opt[j+1] + A[j+1][i] * x[i];
      y_opt[j+2] = y_opt[j+2] + A[i][j+2] * x[i];
      z_opt[j+2] = z_opt[j+2] + A[j+2][i] * x[i];
      y_opt[j+3] = y_opt[j+3] + A[i][j+3] * x[i];
      z_opt[j+3] = z_opt[j+3] + A[j+3][i] * x[i];
    }
  }
}



void optimized3(double** __restrict__ A, double* __restrict__ x, double* __restrict__ y_opt, double* __restrict__ z_opt) {
  
  
  #pragma GCC ivdep
  
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j=j+8) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      z_opt[j] = z_opt[j] + A[j][i] * x[i];
      y_opt[j+1] = y_opt[j+1] + A[i][j+1] * x[i];
      z_opt[j+1] = z_opt[j+1] + A[j+1][i] * x[i];
      y_opt[j+2] = y_opt[j+2] + A[i][j+2] * x[i];
      z_opt[j+2] = z_opt[j+2] + A[j+2][i] * x[i];
      y_opt[j+3] = y_opt[j+3] + A[i][j+3] * x[i];
      z_opt[j+3] = z_opt[j+3] + A[j+3][i] * x[i];
      y_opt[j+4] = y_opt[j+4] + A[i][j+4] * x[i];
      z_opt[j+4] = z_opt[j+4] + A[j+4][i] * x[i];
      y_opt[j+5] = y_opt[j+5] + A[i][j+5] * x[i];
      z_opt[j+5] = z_opt[j+5] + A[j+5][i] * x[i];
      y_opt[j+6] = y_opt[j+6] + A[i][j+6] * x[i];
      z_opt[j+6] = z_opt[j+6] + A[j+6][i] * x[i];
      y_opt[j+7] = y_opt[j+7] + A[i][j+7] * x[i];
      z_opt[j+7] = z_opt[j+7] + A[j+7][i] * x[i];
    }
  }
}

void optimized4(double** __restrict__ A, double* __restrict__ x, double* __restrict__ y_opt, double* __restrict__ z_opt) {
  
  #pragma GCC ivdep
  
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j=j+16) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      z_opt[j] = z_opt[j] + A[j][i] * x[i];
      y_opt[j+1] = y_opt[j+1] + A[i][j+1] * x[i];
      z_opt[j+1] = z_opt[j+1] + A[j+1][i] * x[i];
      y_opt[j+2] = y_opt[j+2] + A[i][j+2] * x[i];
      z_opt[j+2] = z_opt[j+2] + A[j+2][i] * x[i];
      y_opt[j+3] = y_opt[j+3] + A[i][j+3] * x[i];
      z_opt[j+3] = z_opt[j+3] + A[j+3][i] * x[i];
      y_opt[j+4] = y_opt[j+4] + A[i][j+4] * x[i];
      z_opt[j+4] = z_opt[j+4] + A[j+4][i] * x[i];
      y_opt[j+5] = y_opt[j+5] + A[i][j+5] * x[i];
      z_opt[j+5] = z_opt[j+5] + A[j+5][i] * x[i];
      y_opt[j+6] = y_opt[j+6] + A[i][j+6] * x[i];
      z_opt[j+6] = z_opt[j+6] + A[j+6][i] * x[i];
      y_opt[j+7] = y_opt[j+7] + A[i][j+7] * x[i];
      z_opt[j+7] = z_opt[j+7] + A[j+7][i] * x[i];
      y_opt[j+8] = y_opt[j+8] + A[i][j+8] * x[i];
      z_opt[j+8] = z_opt[j+8] + A[j+8][i] * x[i];
      y_opt[j+9] = y_opt[j+9] + A[i][j+9] * x[i];
      z_opt[j+9] = z_opt[j+9] + A[j+9][i] * x[i];
      y_opt[j+10] = y_opt[j+10] + A[i][j+10] * x[i];
      z_opt[j+10] = z_opt[j+10] + A[j+10][i] * x[i];
      y_opt[j+11] = y_opt[j+11] + A[i][j+11] * x[i];
      z_opt[j+11] = z_opt[j+11] + A[j+11][i] * x[i];
      y_opt[j+12] = y_opt[j+12] + A[i][j+12] * x[i];
      z_opt[j+12] = z_opt[j+12] + A[j+12][i] * x[i];
      y_opt[j+13] = y_opt[j+13] + A[i][j+13] * x[i];
      z_opt[j+13] = z_opt[j+13] + A[j+13][i] * x[i];
      y_opt[j+14] = y_opt[j+14] + A[i][j+14] * x[i];
      z_opt[j+14] = z_opt[j+14] + A[j+14][i] * x[i];
      y_opt[j+15] = y_opt[j+15] + A[i][j+15] * x[i];
      z_opt[j+15] = z_opt[j+15] + A[j+15][i] * x[i];
    }
  }
}


void optimized5(double** __restrict__ A, double* __restrict__ x, double* __restrict__ y_opt, double* __restrict__ z_opt) {
  
  #pragma GCC ivdep
  
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j=j+32) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      z_opt[j] = z_opt[j] + A[j][i] * x[i];
      y_opt[j+1] = y_opt[j+1] + A[i][j+1] * x[i];
      z_opt[j+1] = z_opt[j+1] + A[j+1][i] * x[i];
      y_opt[j+2] = y_opt[j+2] + A[i][j+2] * x[i];
      z_opt[j+2] = z_opt[j+2] + A[j+2][i] * x[i];
      y_opt[j+3] = y_opt[j+3] + A[i][j+3] * x[i];
      z_opt[j+3] = z_opt[j+3] + A[j+3][i] * x[i];
      y_opt[j+4] = y_opt[j+4] + A[i][j+4] * x[i];
      z_opt[j+4] = z_opt[j+4] + A[j+4][i] * x[i];
      y_opt[j+5] = y_opt[j+5] + A[i][j+5] * x[i];
      z_opt[j+5] = z_opt[j+5] + A[j+5][i] * x[i];
      y_opt[j+6] = y_opt[j+6] + A[i][j+6] * x[i];
      z_opt[j+6] = z_opt[j+6] + A[j+6][i] * x[i];
      y_opt[j+7] = y_opt[j+7] + A[i][j+7] * x[i];
      z_opt[j+7] = z_opt[j+7] + A[j+7][i] * x[i];
      y_opt[j+8] = y_opt[j+8] + A[i][j+8] * x[i];
      z_opt[j+8] = z_opt[j+8] + A[j+8][i] * x[i];
      y_opt[j+9] = y_opt[j+9] + A[i][j+9] * x[i];
      z_opt[j+9] = z_opt[j+9] + A[j+9][i] * x[i];
      y_opt[j+10] = y_opt[j+10] + A[i][j+10] * x[i];
      z_opt[j+10] = z_opt[j+10] + A[j+10][i] * x[i];
      y_opt[j+11] = y_opt[j+11] + A[i][j+11] * x[i];
      z_opt[j+11] = z_opt[j+11] + A[j+11][i] * x[i];
      y_opt[j+12] = y_opt[j+12] + A[i][j+12] * x[i];
      z_opt[j+12] = z_opt[j+12] + A[j+12][i] * x[i];
      y_opt[j+13] = y_opt[j+13] + A[i][j+13] * x[i];
      z_opt[j+13] = z_opt[j+13] + A[j+13][i] * x[i];
      y_opt[j+14] = y_opt[j+14] + A[i][j+14] * x[i];
      z_opt[j+14] = z_opt[j+14] + A[j+14][i] * x[i];
      y_opt[j+15] = y_opt[j+15] + A[i][j+15] * x[i];
      z_opt[j+15] = z_opt[j+15] + A[j+15][i] * x[i];
      y_opt[j+16] = y_opt[j+16] + A[i][j+16] * x[i];
      z_opt[j+16] = z_opt[j+16] + A[j+16][i] * x[i];
      y_opt[j+17] = y_opt[j+17] + A[i][j+17] * x[i];
      z_opt[j+17] = z_opt[j+17] + A[j+17][i] * x[i];
      y_opt[j+18] = y_opt[j+18] + A[i][j+18] * x[i];
      z_opt[j+18] = z_opt[j+18] + A[j+18][i] * x[i];
      y_opt[j+19] = y_opt[j+19] + A[i][j+19] * x[i];
      z_opt[j+19] = z_opt[j+19] + A[j+19][i] * x[i];
      y_opt[j+20] = y_opt[j+20] + A[i][j+20] * x[i];
      z_opt[j+20] = z_opt[j+20] + A[j+20][i] * x[i];
      y_opt[j+21] = y_opt[j+21] + A[i][j+21] * x[i];
      z_opt[j+21] = z_opt[j+21] + A[j+21][i] * x[i];
      y_opt[j+22] = y_opt[j+22] + A[i][j+22] * x[i];
      z_opt[j+22] = z_opt[j+22] + A[j+22][i] * x[i];
      y_opt[j+23] = y_opt[j+23] + A[i][j+23] * x[i];
      z_opt[j+23] = z_opt[j+23] + A[j+23][i] * x[i];
      y_opt[j+24] = y_opt[j+24] + A[i][j+24] * x[i];
      z_opt[j+24] = z_opt[j+24] + A[j+24][i] * x[i];
      y_opt[j+25] = y_opt[j+25] + A[i][j+25] * x[i];
      z_opt[j+25] = z_opt[j+25] + A[j+25][i] * x[i];
      y_opt[j+26] = y_opt[j+26] + A[i][j+26] * x[i];
      z_opt[j+26] = z_opt[j+26] + A[j+26][i] * x[i];
      y_opt[j+27] = y_opt[j+27] + A[i][j+27] * x[i];
      z_opt[j+27] = z_opt[j+27] + A[j+27][i] * x[i];
      y_opt[j+28] = y_opt[j+28] + A[i][j+28] * x[i];
      z_opt[j+28] = z_opt[j+28] + A[j+28][i] * x[i];
      y_opt[j+29] = y_opt[j+29] + A[i][j+29] * x[i];
      z_opt[j+29] = z_opt[j+29] + A[j+29][i] * x[i];
      y_opt[j+30] = y_opt[j+30] + A[i][j+30] * x[i];
      z_opt[j+30] = z_opt[j+30] + A[j+30][i] * x[i];
      y_opt[j+31] = y_opt[j+31] + A[i][j+31] * x[i];
      z_opt[j+31] = z_opt[j+31] + A[j+31][i] * x[i];
    }
  }
}


void optimized6(double** __restrict__ A, double* __restrict__ x, double* __restrict__ y_opt, double* __restrict__ z_opt) {

  
  
  #pragma GCC ivdep
  
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
    }
  }

  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      z_opt[j] = z_opt[j] + A[j][i] * x[i];
    }
  }

}



void optimized7(double** __restrict__ A, double* __restrict__ x, double* __restrict__ y_opt, double* __restrict__ z_opt) {

  
  
  #pragma GCC ivdep
  
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j=j+8) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      y_opt[j+1] = y_opt[j+1] + A[i][j+1] * x[i];
      y_opt[j+2] = y_opt[j+2] + A[i][j+2] * x[i];
      y_opt[j+3] = y_opt[j+3] + A[i][j+3] * x[i];
      y_opt[j+4] = y_opt[j+4] + A[i][j+4] * x[i];
      y_opt[j+5] = y_opt[j+5] + A[i][j+5] * x[i];
      y_opt[j+6] = y_opt[j+6] + A[i][j+6] * x[i];
      y_opt[j+7] = y_opt[j+7] + A[i][j+7] * x[i];
    }
  }

  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i=i+8) {
      z_opt[j] = z_opt[j] + A[j][i] * x[i];
      z_opt[j] = z_opt[j] + A[j][i+1] * x[i+1];
      z_opt[j] = z_opt[j] + A[j][i+2] * x[i+2];
      z_opt[j] = z_opt[j] + A[j][i+3] * x[i+3];
      z_opt[j] = z_opt[j] + A[j][i+4] * x[i+4];
      z_opt[j] = z_opt[j] + A[j][i+5] * x[i+5];
      z_opt[j] = z_opt[j] + A[j][i+6] * x[i+6];
      z_opt[j] = z_opt[j] + A[j][i+7] * x[i+7];

    }
  }
}

void optimized9(double** __restrict__ A, double* __restrict__ x, double* __restrict__ y_opt, double* __restrict__ z_opt) {

  
  
  #pragma GCC ivdep
  
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j=j+4) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      y_opt[j+1] = y_opt[j+1] + A[i][j+1] * x[i];
      y_opt[j+2] = y_opt[j+2] + A[i][j+2] * x[i];
      y_opt[j+3] = y_opt[j+3] + A[i][j+3] * x[i];
    }
  }

  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i=i+4) {
      z_opt[j] = z_opt[j] + A[j][i] * x[i];
      z_opt[j] = z_opt[j] + A[j][i+1] * x[i+1];
      z_opt[j] = z_opt[j] + A[j][i+2] * x[i+2];
      z_opt[j] = z_opt[j] + A[j][i+3] * x[i+3];
    }
  }
}


void avx_version(double** __restrict__ A, double* __restrict__ x, double* __restrict__ y_opt, double* __restrict__ z_opt) {

  __m256d rA, rX, rY_opt,rZ_opt ,ri;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j=j+4) {
      rY_opt =  _mm256_loadu_pd(&y_opt[j]);
      rX = _mm256_set1_pd(x[i]);
      rA = _mm256_loadu_pd(&A[i][j]);
      ri = _mm256_mul_pd(rA,rX);
      rY_opt = _mm256_add_pd(rY_opt,ri);
       
     _mm256_storeu_pd(&y_opt[j],rY_opt);
    }
  }

  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i=i+4) {
      rX= _mm256_load_pd(&x[i]);
      rA= _mm256_load_pd(&A[j][i]);
      rZ_opt= _mm256_set1_pd(z_opt[j]);
      // rZ_opt = rZ_opt + rA*rX;
      ri = _mm256_mul_pd(rA,rX);
      rZ_opt = _mm256_add_pd(rZ_opt,ri);
      rZ_opt = _mm256_hadd_pd(rZ_opt,rZ_opt);
      rZ_opt = _mm256_hadd_pd(rZ_opt, rZ_opt);
      __m256d rZ_opt_flip = _mm256_permute2f128_pd(rZ_opt, rZ_opt , 1);
      rZ_opt = _mm256_add_pd(rZ_opt,rZ_opt_flip); 
      z_opt[j] = _mm256_cvtsd_f64(rZ_opt);

    }
  }
}

int main() {
  double clkbegin, clkend;
  double t;

  int i, j, it;
  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(5);

  double** A;
  A = new double*[N];
  for (int i = 0; i < N; i++) {
    A[i] = new double[N];
  }

  double *x, *y_ref, *z_ref, *y_opt, *z_opt;
  x = new double[N];
  y_ref = new double[N];
  z_ref = new double[N];
  y_opt = new double[N];
  z_opt = new double[N];

  for (i = 0; i < N; i++) {
    x[i] = i;
    y_ref[i] = 1.0;
    y_opt[i] = 1.0;
    z_ref[i] = 2.0;
    z_opt[i] = 2.0;
    for (j = 0; j < N; j++) {
      A[i][j] = (i + 2.0 * j) / (2.0 * N);
    }
  }

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    reference(A, x, y_ref, z_ref);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Reference Version: Matrix Size = " << N << ", " << 4.0 * 1e-9 * N * N * Niter / t
       << " GFLOPS; Time = " << t / Niter << " sec\n";

  clkbegin = rtclock();
  cout<<"########LOOP PERMUTATION########"<<endl;
  for (it = 0; it < Niter; it++) {
    optimized1(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);

  //Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  clkbegin = rtclock();
  cout<<"########LOOP FISSION########"<<endl;
  for (it = 0; it < Niter; it++) {
    optimized2(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);


  //Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  clkbegin = rtclock();
  cout<<"########LOOP UNROLLING 4########"<<endl;
  for (it = 0; it < Niter; it++) {
    optimized8(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);
  
  //Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  clkbegin = rtclock();
  cout<<"########LOOP UNROLLING 8########"<<endl;
  for (it = 0; it < Niter; it++) {
    optimized3(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);



  //Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  clkbegin = rtclock();
  cout<<"########LOOP UNROLLING 16########"<<endl;
  for (it = 0; it < Niter; it++) {
    optimized4(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);


  //Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  clkbegin = rtclock();
  cout<<"########LOOP UNROLLING 32########"<<endl;

  for (it = 0; it < Niter; it++) {
    optimized5(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);

  //Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  clkbegin = rtclock();
  cout<<"########LOOP FISSION + PERMUATION########"<<endl;
  for (it = 0; it < Niter; it++) {
    optimized6(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);

  //Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  clkbegin = rtclock();
  cout<<"########LOOP FISSION + PERMUATION + UNROLLING4########"<<endl;
  for (it = 0; it < Niter; it++) {
    optimized9(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);

  // Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  
  //Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  clkbegin = rtclock();
  cout<<"########LOOP FISSION + PERMUATION + UNROLLING8########"<<endl;
  for (it = 0; it < Niter; it++) {
    optimized7(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);

  // Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // Another optimized version possibly

  // Version with intinsics

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    avx_version(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Intrinsics Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);

  return EXIT_SUCCESS;
}
