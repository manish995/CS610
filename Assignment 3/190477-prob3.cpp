// Compile: g++ -O2 -fopenmp -o problem3 problem3.cpp
// Execute: ./problem3

#include <cassert>
#include <iostream>
#include <omp.h>
#include <immintrin.h>

#define N (1 << 12)
#define ITER 100

using std::cout;
using std::endl;

void check_result(uint32_t** w_ref, uint32_t** w_opt) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      assert(w_ref[i][j] == w_opt[i][j]);
    }
  }
  cout << "No differences found between base and test versions\n";
}

void reference(uint32_t** A) {
  int i, j, k;
  for (k = 0; k < ITER; k++) {
    for (i = 1; i < N; i++) {
      for (j = 0; j < (N - 1); j++) {
        A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];//
      }
    }
  }
}

// TODO: MAKE YOUR CHANGES TO OPTIMIZE THIS FUNCTION
void omp_version1(uint32_t** A) {
  //simply parallelising the jth loop with kij permutation
  int i, j, k;
 
 #pragma omp parallel num_threads(2) private(i,j,k)
	  for (k = 0; k < ITER; k++) 
	  {
	    for (i = 1; i < N; i++) 
	    {
	    #pragma omp for nowait
	      for (j = 0; j < (N-1); j+=1) 
	      {
          A[i][j+1] = A[i - 1][j+1] + A[i][j+1];
      	}
	    }
	  }
}


void omp_version2(uint32_t** A) {
  int i, j, k;
 //jki permutation with j loop parallel
 #pragma omp parallel num_threads(4) private(i,j,k)
    
  #pragma omp for nowait
    for (j = 0; j < (N-1); j+=1)
	  {
	   for (k = 0; k < ITER; k++) 
	    {
	       for (i = 1; i < N; i++) 
	      {
          A[i][j+1] = A[i - 1][j+1] + A[i][j+1];
        }
	    }
	  }
    
}




void omp_version3(uint32_t** A) {
  int i, j, k;
 //jki permutation with loop unrolling
 #pragma omp parallel num_threads(4) private(i,j,k)
    
  #pragma omp for nowait
    for (j = 0; j < (N-1); j+=1)
	  {
	   for (k = 0; k < ITER; k++) 
	    {
	       for (i = 1; i < N-8; i=i+8) 
	      {
          A[i][j+1] = A[i - 1][j+1] + A[i][j+1];
          A[i+1][j+1] = A[i][j+1] + A[i+1][j+1];
          A[i+2][j+1] = A[i+1][j+1] + A[i+2][j+1];
          A[i+3][j+1] = A[i+2][j+1] + A[i+3][j+1];
          A[i+4][j+1] = A[i+3][j+1] + A[i+4][j+1];
          A[i+5][j+1] = A[i+4][j+1] + A[i+5][j+1];
          A[i+6][j+1] = A[i+5][j+1] + A[i+6][j+1];
          A[i+7][j+1] = A[i+6][j+1] + A[i+7][j+1];
      	}
        for (i = N-7; i < N;i=i+8) 
	      {
          A[i][j+1] = A[i - 1][j+1] + A[i][j+1];
          A[i+1][j+1] = A[i][j+1] + A[i+1][j+1];
          A[i+2][j+1] = A[i+1][j+1] + A[i+2][j+1];
          A[i+3][j+1] = A[i+2][j+1] + A[i+3][j+1];
          A[i+4][j+1] = A[i+3][j+1] + A[i+4][j+1];
          A[i+5][j+1] = A[i+4][j+1] + A[i+5][j+1];
          A[i+6][j+1] = A[i+5][j+1] + A[i+6][j+1];
          // A[i+7][j+1] = A[i+6][j+1] + A[i+7][j+1];
      	}

	    }
	  }
    
  
}

void omp_version4(uint32_t** A) {
  int i, j, k;
 
 //Kij permuation with loop unrolling
 #pragma omp parallel num_threads(4) private(i,j,k)
	  for (k = 0; k < ITER; k++) 
	  {
	    for (i = 1; i < N; i++) 
	    {
	    #pragma omp for nowait
	      for (j = 0; j < (N-1); j+=8) 
	      {
          A[i][j+1] = A[i - 1][j+1] + A[i][j+1];
          A[i][j+2] = A[i - 1][j+2] + A[i][j+2];
          A[i][j+3] = A[i - 1][j+3] + A[i][j+3];
          A[i][j+4] = A[i - 1][j+4] + A[i][j+4];
          A[i][j+5] = A[i - 1][j+5] + A[i][j+5];
          A[i][j+6] = A[i - 1][j+6] + A[i][j+6];
          A[i][j+7] = A[i - 1][j+7] + A[i][j+7];
          A[i][j+8] = A[i - 1][j+8] + A[i][j+8];
          // A[i][j+9] = A[i - 1][j+9] + A[i][j+9];
          // A[i][j+10] = A[i - 1][j+10] + A[i][j+10];
          // A[i][j+11] = A[i - 1][j+11] + A[i][j+11];
          // A[i][j+12] = A[i - 1][j+12] + A[i][j+12];
          // A[i][j+13] = A[i - 1][j+13] + A[i][j+13];
          // A[i][j+14] = A[i - 1][j+14] + A[i][j+14];
          // A[i][j+15] = A[i - 1][j+15] + A[i][j+15];
          // A[i][j+16] = A[i - 1][j+16] + A[i][j+16];
      	}
	    }
	  }
}

void omp_version5(uint32_t** A) {
  int i, j, k;
 //kji permuation with j loop parallel
 #pragma omp parallel num_threads(5) private(i,j,k)
 {
	  for (k = 0; k < ITER; k++) 
	  {
	    #pragma omp for nowait
	      for (j = 0; j < (N-1); j++) 
	    {
        for (i = 1; i < N; i++) 
	      {
          A[i][j+1] = A[i - 1][j+1] + A[i][j+1];
      	}
	    }
	  }
 }
}





int main() {
  uint32_t** A_ref = new uint32_t*[N];
  for (int i = 0; i < N; i++) {
    A_ref[i] = new uint32_t[N];
  }

  uint32_t** A_omp = new uint32_t*[N];
  for (int i = 0; i < N; i++) {
    A_omp[i] = new uint32_t[N];
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_ref[i][j] = i + j + 1;
      A_omp[i][j] = i + j + 1;
    }
  }

  double start = omp_get_wtime();
  reference(A_ref);
  double end = omp_get_wtime();
  cout << "Time for reference version: " << end - start << " seconds\n";

  start = omp_get_wtime();
  omp_version1(A_omp);
  end = omp_get_wtime();
  check_result(A_ref, A_omp);
  cout << "Version1: Time with OpenMP: " << end - start << " seconds\n";

  // Reset
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_omp[i][j] = i + j + 1;
    }
  }

  start = omp_get_wtime();
  omp_version2(A_omp);
  end = omp_get_wtime();
  check_result(A_ref, A_omp);
  cout << "Version2: Time with OpenMP: " << end - start << " seconds\n";


  // Reset
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_omp[i][j] = i + j + 1;
    }
  }

  start = omp_get_wtime();
  omp_version3(A_omp);
  end = omp_get_wtime();
  check_result(A_ref, A_omp);
  cout << "Version3: Time with OpenMP: " << end - start << " seconds\n";


  // Reset
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_omp[i][j] = i + j + 1;
    }
  }

  start = omp_get_wtime();
  omp_version4(A_omp);
  end = omp_get_wtime();
  check_result(A_ref, A_omp);
  cout << "Version4: Time with OpenMP: " << end - start << " seconds\n";

  // // Reset
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_omp[i][j] = i + j + 1;
    }
  }

  start = omp_get_wtime();
  omp_version5(A_omp);
  end = omp_get_wtime();
  check_result(A_ref, A_omp);
  cout << "Version5: Time with OpenMP: " << end - start << " seconds\n";

  // Another optimized version possibly

  return EXIT_SUCCESS;
}
