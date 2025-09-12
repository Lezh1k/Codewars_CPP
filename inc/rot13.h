#ifndef ROT13_H
#define ROT13_H

#include <stddef.h>
// AUX.h
#include <chrono>
#include <functional>
#include <iostream>
#include <string>

struct func_profiler {
  std::string m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_start;

  func_profiler() = delete;
  func_profiler(const std::string &name)
      : m_name(name), m_start(std::chrono::high_resolution_clock::now()) {}
  ~func_profiler() {
    auto dur = std::chrono::high_resolution_clock::now() - m_start;
    std::cout
        << m_name << ": "
        << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count()
        << " ms.\n";
  }
};
size_t get_cpu_count();
void run_in_parallel(std::function<void(size_t)> pf_func);
//////////////////////////////////////////////////////////////

char rot13_dummy(char c);
char rot13_opt1(char c);

void rot13_naive(char *str, size_t n);
void rot13_naive_parallel(char *str, size_t n);

void rot13_opt1(char *str, size_t n);
void rot13_opt1_parallel(char *str, size_t n);

void rot13_lut(char *str, size_t n);
void rot13_lut_parallel(char *str, size_t n);

void rot13_sse(char *str, size_t n);
void rot13_sse_parallel(char *str, size_t n);

void rot13_sse_prefetch(char *str, size_t n);
void rot13_sse_prefetch_parallel(char *str, size_t n);

void rot13_avx2(char *str, size_t n);
void rot13_avx2_parallel(char *str, size_t n);

void cuda_rot13(char *str, size_t n);
void cuda_rot13_vect(char *str, size_t n);

#if (__AVX512_ENABLED)
void rot13_avx512(char *str, size_t n);
#endif

#endif
