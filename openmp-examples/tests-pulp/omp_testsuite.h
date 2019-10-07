/* Adapted from 
/* Global headerfile of the OpenMP Testsuite */

#ifndef OMP_TESTSUITE_H
#define OMP_TESTSUITE_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* General                                                */
/**********************************************************/
#define LOOPCOUNT  50 /* Number of iterations to slit amongst threads */
#define REPETITIONS 1 /* Number of times to run each test */

/* following times are in seconds */
#define SLEEPTIME 1

/* Definitions for tasks                                  */
/**********************************************************/
#define NUM_TASKS 25
#define MAX_TASKS_PER_THREAD 5

void fprintf(char *c, char *output)

#endif
