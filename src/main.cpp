#include <emmintrin.h>
#include <gtest/gtest.h>
#include <immintrin.h>
#include <xmmintrin.h>

#include <chrono>
#include <functional>
#include <thread>

#include "rot13.cuh"

#ifdef _UNIT_TESTS_
int main_tests(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
//////////////////////////////////////////////////////////////

static char rot13(char c);
static void rot13_naive(char *str, size_t n);
static void rot13_sse(char *str, size_t n);

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
//////////////////////////////////////////////////////////////

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

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  cuda_hello();
  return 0;

  static const std::string pattern =
      "12345 abcdefghijklmnopqrstuvwxyz "
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ 09876 !!! @`[{";

  const int repeats = 100000000;
  const int iterations = 1;
  const size_t buff_len = pattern.size() * repeats + 1;
  const size_t cpu_n = std::thread::hardware_concurrency();

  // SSE batches and offsets
  size_t sse_bs = buff_len / cpu_n;
  while (sse_bs % 16) {
    --sse_bs;
  }

  std::vector<size_t> lst_sse_batch_size(cpu_n, sse_bs);
  const size_t sse_remind = buff_len - (sse_bs * cpu_n);
  lst_sse_batch_size.back() += sse_remind;

  std::vector<size_t> lst_sse_batch_offset(cpu_n, 0);
  for (size_t i = 1; i < lst_sse_batch_size.size(); ++i) {
    lst_sse_batch_offset[i] =
        lst_sse_batch_offset[i - 1] + lst_sse_batch_size[i - 1];
  }
  //////////////////////////////////////////////////////////////

  // naive batches and offsets
  size_t naive_bs = buff_len / cpu_n;
  std::vector<size_t> lst_naive_batch_size(cpu_n, naive_bs);
  const size_t naive_remind = buff_len - (naive_bs * cpu_n);
  for (size_t i = 0; i < naive_remind; ++i) {
    ++lst_naive_batch_size[i];
  }
  std::vector<size_t> lst_naive_batch_offset(cpu_n, 0);
  for (size_t i = 1; i < lst_naive_batch_size.size(); ++i) {
    lst_naive_batch_offset[i] =
        lst_naive_batch_offset[i - 1] + lst_naive_batch_size[i - 1];
  }
  //////////////////////////////////////////////////////////////

  std::string orig_big_input;
  orig_big_input.reserve(buff_len);

  // Fill original buffer
  for (int i = 0; i < repeats; ++i) {
    orig_big_input += pattern;
  }

  // Working buffer reused per iteration
  char *work = strdup(orig_big_input.c_str());
  if (!work) {
    std::cerr << "malloc work failed\n";
    return 1;
  }

  std::vector<benchmark> benchmarks = {
      benchmark("naive", [work, buff_len]() { rot13_naive(work, buff_len); }),
      benchmark("sse", [work, buff_len]() { rot13_sse(work, buff_len); }),
      benchmark(
          "naive parallel",
          [work, &lst_naive_batch_size, &lst_naive_batch_offset, cpu_n]() {
            std::vector<std::thread> threads(cpu_n);
            for (size_t i = 0; i < cpu_n; ++i) {
              threads[i] = std::thread(
                  [work, &lst_naive_batch_size, &lst_naive_batch_offset, i]() {
                    rot13_naive(work + lst_naive_batch_offset[i],
                                lst_naive_batch_size[i]);
                  });
              cpu_set_t cpuset;
              CPU_ZERO(&cpuset);
              CPU_SET(i, &cpuset);
              int rc = pthread_setaffinity_np(threads[i].native_handle(),
                                              sizeof(cpu_set_t), &cpuset);
              if (rc != 0) {
                std::cerr << "Error calling pthread_setaffinity_np: " << rc
                          << "\n";
              }
            }

            for (auto &t : threads) {
              if (!t.joinable())
                continue;
              t.join();
            }
          }),
      benchmark("sse parallel",
                [work, &lst_sse_batch_size, &lst_sse_batch_offset, cpu_n]() {
                  std::vector<std::thread> threads(cpu_n);
                  for (size_t i = 0; i < cpu_n; ++i) {
                    threads[i] = std::thread([work, &lst_sse_batch_size,
                                              &lst_sse_batch_offset, i]() {
                      rot13_sse(work + lst_sse_batch_offset[i],
                                lst_sse_batch_size[i]);
                    });
                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    CPU_SET(i, &cpuset);
                    int rc = pthread_setaffinity_np(threads[i].native_handle(),
                                                    sizeof(cpu_set_t), &cpuset);
                    if (rc != 0) {
                      std::cerr
                          << "Error calling pthread_setaffinity_np: " << rc
                          << "\n";
                    }
                  }

                  for (auto &t : threads) {
                    if (!t.joinable())
                      continue;
                    t.join();
                  }
                }),
  };

  std::cout << "processing text: " << buff_len / (1024 * 1024) << " MB\n";

  for (benchmark &b : benchmarks) {
    func_profiler p(b.m_name);
    for (int i = 0; i < iterations; ++i) {
      memcpy(work, orig_big_input.c_str(), buff_len);
      b.m_func();

      // Touch the data so the call can't be optimized out
      // (simple byte sum; not performance-critical)
      uint64_t sum = 0;
      for (size_t i = 0; i < buff_len; ++i)
        sum += static_cast<uint8_t>(work[i]);
      b.m_sink += sum;
    }

    std::cout << "(ignore) sink " << b.m_name << " : " << b.m_sink << "\n";
  }

  free(work); // because of strdup
  return 0;
}
//////////////////////////////////////////////////////////////

char rot13(char c) {
  char cl = c | 0x20; // to lower
  int8_t is_alpha = (uint8_t)(cl - 'a') <= 'z' - 'a';
  int8_t offset = 13 - 26 * (cl > 'm');
  c += is_alpha * offset;
  return c;
}
//////////////////////////////////////////////////////////////

void rot13_naive(char *str, size_t n) {
  size_t i = 0;
  char *s = str;
  for (; *s && i < n; ++s, ++i) {
    *s = rot13(*s);
  }
}
//////////////////////////////////////////////////////////////

void rot13_sse(char *str, size_t n) {
  alignas(16) const uint8_t msk_20_data[16] = {
      0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
      0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20};
  alignas(16) const uint8_t msk_a_data[16] = {
      'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
      'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
  };
  alignas(16) const uint8_t msk_z_data[16] = {
      'z' + 1, 'z' + 1, 'z' + 1, 'z' + 1, 'z' + 1, 'z' + 1, 'z' + 1, 'z' + 1,
      'z' + 1, 'z' + 1, 'z' + 1, 'z' + 1, 'z' + 1, 'z' + 1, 'z' + 1, 'z' + 1,
  };
  alignas(16) const uint8_t msk_m_data[16] = {
      'm' + 1, 'm' + 1, 'm' + 1, 'm' + 1, 'm' + 1, 'm' + 1, 'm' + 1, 'm' + 1,
      'm' + 1, 'm' + 1, 'm' + 1, 'm' + 1, 'm' + 1, 'm' + 1, 'm' + 1, 'm' + 1,
  };
  alignas(16) const uint8_t msk_13_data[16] = {
      13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
  };
  alignas(16) const uint8_t msk_26_data[16] = {
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
  };

  uintptr_t p_str = (uintptr_t)str;
  uintptr_t p_aligned = (p_str + 15) & ~15;

  // for analigned data we use naive approach
  char *s = str;
  size_t i = 0;
  for (; s != (char *)p_aligned && i < n; ++s, ++i) {
    *s = rot13(*s);
  }

  // sse while possible
  __m128i zero = _mm_setzero_si128();
  __m128i ffff = _mm_cmpeq_epi8(zero, zero);

  for (; i < n; s += 16, i += 16) {
    __m128i orig = _mm_load_si128((__m128i *)s);
    __m128i eq_zero = _mm_cmpeq_epi8(orig, zero);
    int mask = _mm_movemask_epi8(eq_zero);
    if (mask) {
      break;
    }

    __m128i msk_20 = _mm_load_si128((__m128i *)msk_20_data);
    __m128i msk_a = _mm_load_si128((__m128i *)msk_a_data);
    __m128i msk_m = _mm_load_si128((__m128i *)msk_m_data);
    __m128i msk_z = _mm_load_si128((__m128i *)msk_z_data);

    __m128i lower_case = _mm_or_si128(orig, msk_20);
    __m128i greater_than_a = _mm_cmpgt_epi8(msk_a, lower_case);
    greater_than_a =
        _mm_xor_si128(greater_than_a, ffff); // 0xff on places greater than a
    __m128i lower_than_z =
        _mm_cmpgt_epi8(msk_z, lower_case); // 0xff on places less than z

    __m128i is_alpha = _mm_and_si128(greater_than_a, lower_than_z);
    __m128i lower_alphas = _mm_and_si128(is_alpha, lower_case);
    __m128i greater_than_m = _mm_cmpgt_epi8(msk_m, lower_alphas);
    greater_than_m = _mm_xor_si128(greater_than_m, ffff);
    greater_than_m = _mm_and_si128(greater_than_m, is_alpha);

    __m128i offset = _mm_load_si128((__m128i *)msk_13_data);
    offset = _mm_and_si128(offset, is_alpha);
    __m128i msk_26 = _mm_load_si128((__m128i *)msk_26_data);
    msk_26 = _mm_and_si128(msk_26, greater_than_m);

    orig = _mm_add_epi8(orig, offset);
    orig = _mm_sub_epi8(orig, msk_26);
    _mm_store_si128((__m128i *)s, orig);
  }

  for (; *s && i < n; ++s, ++i) {
    *s = rot13(*s);
  }
}
//////////////////////////////////////////////////////////////
