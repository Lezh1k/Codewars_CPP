#include <emmintrin.h>
#include <gtest/gtest.h>
#include <immintrin.h>
#include <xmmintrin.h>

#include <chrono>
#include <iomanip>

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

void p128_hex_u8(__m128i in) {
  alignas(16) uint8_t v[16];
  _mm_store_si128((__m128i *)v, in);
  std::cout << "v16_u8: " << std::hex << std::setw(2) << std::setfill('0')
            << v[0] << v[1] << v[2] << v[3] << " | " << v[4] << v[5] << v[6]
            << v[7] << " | " << v[8] << v[9] << v[10] << v[11] << " | " << v[12]
            << v[13] << v[14] << v[15] << "\n";
}
//////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  using namespace std::chrono;
  (void)argc;
  (void)argv;
  static const std::string pattern =
      "12345 abcdefghijklmnopqrstuvwxyz "
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ 09876 !!! @`[{";

  const int repeats = 1000000;
  const int iterations = 100;
  const size_t buf_len = pattern.size() * repeats + 1;
  std::string orig_big_input;
  orig_big_input.reserve(buf_len);

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

  // Accumulators to prevent the compiler from optimizing work away
  // (volatile sink so results are observable)
  volatile uint64_t sink_sse = 0, sink_naive = 0;

  std::cout << "Buffer size: " << buf_len / (1024*1024) << " MBs\n";
  std::cout << "Iterations per function: " << iterations << "\n";

  // Benchmark rot13_sse
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int it = 0; it < iterations; ++it) {
    memcpy(work, orig_big_input.c_str(), buf_len + 1);
    rot13_sse(work, buf_len + 1);
    // Touch the data so the call can't be optimized out
    // (simple byte sum; not performance-critical)
    uint64_t sum = 0;
    for (size_t i = 0; i < buf_len; ++i)
      sum += (unsigned char)work[i];
    sink_sse += sum;
  }
  auto t2 = high_resolution_clock::now();
  auto dur_sse = t2 - t1;
  std::cout << "rot13_sse: " << duration_cast<milliseconds>(dur_sse).count()
            << " ms.\n";

  // Benchmark rot13_naive
  for (int it = 0; it < iterations; ++it) {
    memcpy(work, orig_big_input.c_str(), buf_len + 1);
    rot13_naive(work, buf_len + 1);
    uint64_t sum = 0;
    for (size_t i = 0; i < buf_len; ++i)
      sum += (unsigned char)work[i];
    sink_naive += sum;
  }
  auto t3 = high_resolution_clock::now();
  auto dur_naive = t3 - t2;
  std::cout << "rot13_naive: " << duration_cast<milliseconds>(dur_naive).count()
            << " ms.\n";

  // Benchmark rot13_sse_parallel
#pragma omp parallel for schedule(dynamic)
  for (int it = 0; it < iterations; ++it) {
    memcpy(work, orig_big_input.c_str(), buf_len + 1);
    rot13_sse(work, buf_len + 1);
    uint64_t sum = 0;
    for (size_t i = 0; i < buf_len; ++i)
      sum += (unsigned char)work[i];
    sink_naive += sum;
  }
  auto t4 = high_resolution_clock::now();
  auto dur_sse_parallel = t4 - t3;
  std::cout << "rot13_sse parallel: "
            << duration_cast<milliseconds>(dur_sse_parallel).count()
            << " ms.\n";

  // Benchmark rot13_naive parallel
#pragma omp parallel for schedule(dynamic)
  for (int it = 0; it < iterations; ++it) {
    memcpy(work, orig_big_input.c_str(), buf_len + 1);
    rot13_naive(work, buf_len + 1);
    uint64_t sum = 0;
    for (size_t i = 0; i < buf_len; ++i)
      sum += (unsigned char)work[i];
    sink_naive += sum;
  }
  auto t5 = high_resolution_clock::now();
  auto dur_naive_parallel = t5 - t4;

  std::cout << "rot13_naive parallel: "
            << duration_cast<milliseconds>(dur_naive_parallel).count()
            << " ms.\n";

  // Use sinks (just print to ensure side-effect)
  std::cout << "(ignore) sinks: sse= " << sink_sse << " naive= " << sink_naive;

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
  for (; *s && i < n; ++s) {
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
  for (; s != (char *)p_aligned; ++s) {
    *s = rot13(*s);
  }

  // sse while possible
  __m128i zero = _mm_setzero_si128();
  __m128i ffff = _mm_cmpeq_epi8(zero, zero);

  for (; true; s += 16) {
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

  for (; *s; ++s) {
    *s = rot13(*s);
  }
}
//////////////////////////////////////////////////////////////
