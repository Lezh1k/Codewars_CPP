#include <stdio.h>
#include "rot13.cuh"

__global__ void __cuda_hello(void) {
  printf("Hello from cuda\n");
}

void cuda_hello() {
  __cuda_hello<<<1,1>>>();
}
