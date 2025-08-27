#include "rot13.cuh"
#include <cstdint>

__device__ __forceinline__ char rot13_char(char c) {
  char cl = c | 0x20; // to lower
  int8_t is_alpha = (uint8_t)(cl - 'a') <= 'z' - 'a';
  int8_t offset = 13 - 26 * (cl > 'm');
  c += is_alpha * offset;
  return c;
}

__global__ void __cuda_rot13(char *str, size_t n) {
  size_t i = 0;
  char *s = str;
  for (; *s && i < n; ++s, ++i) {
    *s = rot13_char(*s);
  }
}
//////////////////////////////////////////////////////////////

__global__ void __cuda_rot13_vectorized(char *str, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t i = idx * 4;

  if (i + 3 < n) {
    uchar4 *vec = reinterpret_cast<uchar4 *>(str);
    uchar4 v = vec[idx];

    v.x = rot13_char(v.x);
    v.y = rot13_char(v.y);
    v.z = rot13_char(v.z);
    v.w = rot13_char(v.w);

    vec[idx] = v;
    return;
  }

  // Handle tail (non-multiple-of-4 end part)
  for (int j = 0; j < 4 && (i + j) < n; ++j) {
    str[i + j] = rot13_char(str[i + j]);
  }
}
//////////////////////////////////////////////////////////////

void cuda_rot13_vect(char *str, size_t n) {
  size_t numThreads = 1024; // got from left heel
  size_t numBlocks = (n + 3) / 4 / numThreads;
  char *pd_str;
  cudaMalloc((void **)&pd_str, n);
  cudaMemcpy(pd_str, str, sizeof(char) * n, cudaMemcpyHostToDevice);
  __cuda_rot13_vectorized<<<numBlocks, numThreads>>>(pd_str, n);
  cudaMemcpy(str, pd_str, sizeof(char) * n, cudaMemcpyDeviceToHost);
  cudaFree((void *)pd_str);
}
//////////////////////////////////////////////////////////////

void cuda_rot13(char *str, size_t n) {
  char *pd_str;
  cudaMalloc((void **)&pd_str, n);
  cudaMemcpy(pd_str, str, sizeof(char) * n, cudaMemcpyHostToDevice);
  __cuda_rot13<<<1, 1>>>(pd_str, n);
  cudaMemcpy(str, pd_str, sizeof(char) * n, cudaMemcpyDeviceToHost);
  cudaFree((void *)pd_str);
}
//////////////////////////////////////////////////////////////
