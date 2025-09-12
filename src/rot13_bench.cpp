#include "rot13_bench.h"
#include "rot13.h"

#include <assert.h>
#include <cstring>
#include <emmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>

#include <functional>
#include <iostream>
#include <sys/sysinfo.h>
#include <unistd.h>

void p128_hex_u8(__m128i in) {
  alignas(16) uint8_t v[16];
  _mm_store_si128((__m128i *)v, in);
  printf("%x %x %x %x | %x %x %x %x | %x %x %x %x | %x %x %x %x\n", v[0], v[1],
         v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12],
         v[13], v[14], v[15]);
}
//////////////////////////////////////////////////////////////

struct benchmark {
  std::string m_name;
  std::function<void()> m_func;

  // Accumulators to prevent the compiler from optimizing work away
  // (volatile sink so results are observable)
  volatile uint64_t m_sink;
  benchmark() = delete;
  benchmark(const std::string &name, std::function<void()> &&func)
      : m_name(name), m_func(func), m_sink(0) {}
};
//////////////////////////////////////////////////////////////

int rot13_bench_launch(void) {
  static const std::string pattern =
      "12345 abcdefghijklmnopqrstuvwxyz "
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ 09876 !!! @`[{";

  const int repeats = 2500000;
  const int iterations = 3;
  const size_t buff_len = pattern.size() * repeats + 1;

  std::unique_ptr<char[]> work(new char[buff_len]);
  if (!work) {
    std::cerr << "malloc work failed\n";
    return 1;
  }
  memset(work.get(), 0, buff_len);

  std::vector<benchmark> benchmarks = {
      benchmark("naive",
                [&work, buff_len]() { rot13_naive(work.get(), buff_len); }),
      benchmark("naive opt1",
                [&work, buff_len]() { rot13_opt1(work.get(), buff_len); }),
      benchmark("lut",
                [&work, buff_len]() { rot13_lut(work.get(), buff_len); }),
      benchmark("sse",
                [&work, buff_len]() { rot13_sse(work.get(), buff_len); }),
      benchmark(
          "sse_prefetch",
          [&work, buff_len]() { rot13_sse_prefetch(work.get(), buff_len); }),
      benchmark("avx2",
                [&work, buff_len]() { rot13_avx2(work.get(), buff_len); }),
#if (__AVX512_ENABLED)
      benchmark("avx512",
                [&work, buff_len]() { rot13_avx512(work.get(), buff_len); }),
#endif
      benchmark(
          "naive parallel",
          [&work, buff_len]() { rot13_naive_parallel(work.get(), buff_len); }),
      benchmark(
          "naive opt parallel",
          [&work, buff_len]() { rot13_opt1_parallel(work.get(), buff_len); }),
      benchmark(
          "lut parallel",
          [&work, buff_len]() { rot13_lut_parallel(work.get(), buff_len); }),
      benchmark(
          "sse parallel",
          [&work, buff_len]() { rot13_sse_parallel(work.get(), buff_len); }),
      benchmark("sse prefetch parallel",
                [&work, buff_len]() {
                  rot13_sse_prefetch_parallel(work.get(), buff_len);
                }),
      benchmark(
          "avx2 parallel",
          [&work, buff_len]() { rot13_avx2_parallel(work.get(), buff_len); }),

      benchmark("cuda",
                [&work, buff_len]() { cuda_rot13(work.get(), buff_len); }),
      benchmark("cuda vect",
                [&work, buff_len]() { cuda_rot13_vect(work.get(), buff_len); }),
  };

  std::cout << "processing text: " << (buff_len / (1024 * 1024 * 1024))
            << " GB " << iterations << " times\n";

  for (benchmark &b : benchmarks) {
    for (int i = 0; i < iterations; ++i) {
      // fill work with pattern
      for (size_t offset = 0; offset < buff_len - pattern.size();
           offset += pattern.size()) {
        memcpy(work.get() + offset, pattern.c_str(), pattern.size());
      }

      {
        func_profiler p(b.m_name);
        b.m_func();
      }

      // Touch the data so the call can't be optimized out
      // (simple byte sum; not performance-critical)
      uint64_t sum = 0;
      for (size_t i = 0; i < buff_len; ++i)
        sum += static_cast<uint8_t>(work[i]);
      b.m_sink += sum;
    }

    std::cout << "(ignore) sink " << b.m_name << " : " << b.m_sink << "\n";
  }

  return 0;
}
//////////////////////////////////////////////////////////////
