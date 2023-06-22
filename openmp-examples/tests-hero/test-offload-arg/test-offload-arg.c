/*
 * HERO Test Offload Application
 *
 * Copyright 2018 ETH Zurich, University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <hero-target.h>  // BIGPULP_MEMCPY
#include <stdio.h>        // printf()

#define N 1024
int a[N];
int b[N];
int c[N];
int g[N];

#pragma omp declare target
void test_offload_fn_args(int * __restrict__ a, int * __restrict__ b, int * __restrict__ c) {
  int i;
#pragma omp parallel for
  for(i = 0; i < N; i++)
    c[i] = a[i] * b[i];
}
#pragma omp end declare target

int main(int argc, char *argv[]) {

  int i;
  int error = 0;
#pragma omp parallel for
    for(int i = 0; i < N; i++){
      c[i] = 0;
      a[i] = i;
      b[i] = i % 2;
    }

#pragma omp target map(from: c[0:N]) map(to: a[0:N], b[0:N]) device(BIGPULP_MEMCPY)
  test_offload_fn_args(a,b,c);

#pragma omp parallel for
  for(i = 0; i < N; i++){
    g[i] = a[i] * b[i];
    if(g[i]!=c[i])
      error++;
  }

  printf("Error rate %f%%\n", (error/(float)N)*100.0);
  return error;
}