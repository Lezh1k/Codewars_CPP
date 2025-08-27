#include "rot13.cuh"
#include <cstdint>


__global__ void __cuda_rot13(char *str, size_t n) {
  size_t i = 0;
  char *s = str;
  for (; *s && i < n; ++s, ++i) {
    char cl = *s | 0x20; // to lower
    int8_t is_alpha = (uint8_t)(cl - 'a') <= 'z' - 'a';
    int8_t offset = 13 - 26 * (cl > 'm');
    *s += is_alpha * offset;
  }
}

void cuda_rot13(char *str, size_t n) {
  char *pd_str;
  cudaMalloc((void **)&pd_str, n);
  cudaMemcpy(pd_str, str, sizeof(char) * n, cudaMemcpyHostToDevice);
  __cuda_rot13<<<1, 1>>>(pd_str, n);
  cudaMemcpy(str, pd_str, sizeof(char) * n, cudaMemcpyDeviceToHost);
  cudaFree((void *)pd_str);
}
//////////////////////////////////////////////////////////////
