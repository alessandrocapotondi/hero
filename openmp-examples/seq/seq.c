/* POLYBENCH/GPU-OPENACC
 *
 * This file is a part of the Polybench/GPU-OpenACC suite
 *
 * Contact:
 * William Killian <killian@udel.edu>
 *
 * Copyright 2013, The University of Delaware
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "axpy.h"

/* Array initialization. */
static void init_array(int n, DATA_TYPE POLYBENCH_1D(Y, N, n),
                       DATA_TYPE POLYBENCH_1D(X, N, n)) {
  int i, j;

  for (i = 0; i < N; i++)
    Y[i] = (DATA_TYPE)i;

  for (i = 0; i < N; i++)
    X[i] = (DATA_TYPE)i;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, DATA_TYPE POLYBENCH_1D(Y, N, n)) {
  int i;

  for (i = 0; i < N; i++) {
    fprintf(stderr, DATA_PRINTF_MODIFIER, Y[i]);
  }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_seq(int n, int K, DATA_TYPE POLYBENCH_1D(Y, N, n),
                        DATA_TYPE POLYBENCH_1D(X, N, n)) {
  int i, j, k;
	// Cache line on A9 is 32B, i.e. 8 floats.
	// Cache line on A53/57 is 64B, i.e. 16 floats.
	int stride=16;
//  #pragma omp target data map(tofrom: Y[0:N]) map(to: X[0:N])
//  #pragma omp target
	for (j = 0; j < stride; j++){
      for (i = 0; i < N; i=i+stride) {
        for (k = 0; k < K; k++){
          Y[j+i] += X[i+j];
        }
      }
    }
}


int main(int argc, char **argv) {
#ifdef TIMEPROG
  my_timespec ProgStrt, ProgStop;
  clock_gettime(CLOCK_REALTIME, &ProgStrt);
#endif

  /* Retrieve problem size. */
  int n = N;
  if(argc != 2){
    printf("ERROR: pass kernel relief measure\n");
    return -1;
  }
  int k = atoi(argv[1]);

  /* Variable declaration/allocation. */
  volatile POLYBENCH_1D_ARRAY_DECL(Y, DATA_TYPE, N, n);
	volatile POLYBENCH_1D_ARRAY_DECL(X, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(Y), POLYBENCH_ARRAY(X));


	//init_cpu_perf();
  /* Start timer. */
  //polybench_start_instruments;
#ifdef LOOPFOREVER
	//set_the_flag();
	while(1){
#else
	// Initialize counters to 0
	//wait_for_the_flag();
	//reset_perfcounters();
	#ifdef TIMEKERN
  clock_gettime(CLOCK_REALTIME, &KernStrt);
	#endif
#endif

  kernel_seq(n, k, POLYBENCH_ARRAY(Y), POLYBENCH_ARRAY(X));
#ifdef LOOPFOREVER
	}
#else
	#ifdef TIMEKERN
  	clock_gettime(CLOCK_REALTIME, &KernStop);
	#endif
  // Get counters
	//get_data_cache_misses();
#endif
	/* Stop and print timer. */
  //polybench_stop_instruments;
  //polybench_print_instruments;


  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(Y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(Y);
  POLYBENCH_FREE_ARRAY(X);

#ifdef TIMEPROG
  clock_gettime(CLOCK_REALTIME, &ProgStop);
  accum = (ProgStop.tv_sec - ProgStrt.tv_sec) +
          (ProgStop.tv_nsec - ProgStrt.tv_nsec) / CLOCK_PRECISION;
  printf("%lf,", accum);
#endif

#ifdef TIMEKERN
	eval_kern_time(KernStrt, KernStop);
#endif

  return 0;
}
