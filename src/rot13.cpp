#include "rot13.h"
#include <assert.h>
#include <cstdint>
#include <emmintrin.h>
#include <immintrin.h>
#include <stddef.h>
#include <xmmintrin.h>
#include <sys/sysinfo.h>
#include <thread>

char rot13_dummy(char c) {
  if (c >= 'a' && c <= 'z') {
    c = (c - 'a' + 13) % 26 + 'a';
  } else if (c >= 'A' && c <= 'Z') {
    c = (c - 'A' + 13) % 26 + 'A';
  }
  return c;
}
//////////////////////////////////////////////////////////////

char rot13_opt1(char c) {
  char cl = c | 0x20; // to lower
  uint8_t is_alpha = (uint8_t)(cl - 'a') <= ('z' - 'a');
  uint8_t offset = 13 - 26 * (cl > 'm');
  c += is_alpha * offset;
  return c;
}
//////////////////////////////////////////////////////////////

alignas(64) static unsigned char lut[256];
struct lut_init {
  lut_init() {
    for (int i = 0; i < 256; ++i) {
      lut[i] = rot13_dummy(i);
    }
  }
} lut_init__;
void rot13_lut(char *str, size_t n) {
  for (size_t i = 0; i < n; ++i)
    str[i] = lut[(uint8_t)str[i]];
}

void rot13_lut_parallel(char *str, size_t n) {
  size_t buff_len = n;
  const size_t cpu_n = get_cpu_count();
  size_t naive_bs = buff_len / cpu_n;
  std::vector<size_t> lst_bs(cpu_n, naive_bs);
  const size_t naive_remind = buff_len - (naive_bs * cpu_n);
  for (size_t i = 0; i < naive_remind; ++i) {
    ++lst_bs[i];
  }
  std::vector<size_t> lst_bo(cpu_n, 0);
  for (size_t i = 1; i < lst_bs.size(); ++i) {
    lst_bo[i] = lst_bo[i - 1] + lst_bs[i - 1];
  }

  run_in_parallel([str, &lst_bo, &lst_bs](size_t i) {
    rot13_lut(str + lst_bo[i], lst_bs[i]);
  });
}
//////////////////////////////////////////////////////////////

void rot13_naive(char *str, size_t n) {
  size_t i = 0;
  char *s = str;
  for (; *s && i < n; ++s, ++i) {
    *s = rot13_dummy(*s);
  }
}

void rot13_naive_parallel(char *str, size_t n) {
  size_t buff_len = n;
  const size_t cpu_n = get_cpu_count();
  size_t naive_bs = buff_len / cpu_n;
  std::vector<size_t> lst_bs(cpu_n, naive_bs);
  const size_t naive_remind = buff_len - (naive_bs * cpu_n);
  for (size_t i = 0; i < naive_remind; ++i) {
    ++lst_bs[i];
  }
  std::vector<size_t> lst_bo(cpu_n, 0);
  for (size_t i = 1; i < lst_bs.size(); ++i) {
    lst_bo[i] = lst_bo[i - 1] + lst_bs[i - 1];
  }

  run_in_parallel([str, &lst_bo, &lst_bs](size_t i) {
    rot13_naive(str + lst_bo[i], lst_bs[i]);
  });
}
//////////////////////////////////////////////////////////////

void rot13_opt1(char *str, size_t n) {
  size_t i = 0;
  char *s = str;
  for (; *s && i < n; ++s, ++i) {
    *s = rot13_opt1(*s);
  }
}

void rot13_opt1_parallel(char *str, size_t n) {
  size_t buff_len = n;
  const size_t cpu_n = get_cpu_count();
  size_t naive_bs = buff_len / cpu_n;
  std::vector<size_t> lst_bs(cpu_n, naive_bs);
  const size_t naive_remind = buff_len - (naive_bs * cpu_n);
  for (size_t i = 0; i < naive_remind; ++i) {
    ++lst_bs[i];
  }
  std::vector<size_t> lst_bo(cpu_n, 0);
  for (size_t i = 1; i < lst_bs.size(); ++i) {
    lst_bo[i] = lst_bo[i - 1] + lst_bs[i - 1];
  }

  run_in_parallel([str, &lst_bo, &lst_bs](size_t i) {
    rot13_opt1(str + lst_bo[i], lst_bs[i]);
  });
}
//////////////////////////////////////////////////////////////

void rot13_sse(char *str, size_t n) {
  const __m128i msk_20 = _mm_set1_epi8(0x20);
  const __m128i msk_a = _mm_set1_epi8('a');
  const __m128i msk_m = _mm_set1_epi8('m' + 1);
  const __m128i msk_z = _mm_set1_epi8('z' + 1);
  const __m128i msk_13 = _mm_set1_epi8(13);
  const __m128i msk_26 = _mm_set1_epi8(26);
  const __m128i msk_00 = _mm_setzero_si128();
  const __m128i msk_ff = _mm_cmpeq_epi8(msk_00, msk_00);

  uintptr_t p_str = (uintptr_t)str;
  uintptr_t p_aligned = (p_str + 15) & ~((uintptr_t)15);

  // for analigned data we use naive approach
  char *s = str;
  size_t i = 0;
  for (; s != (char *)p_aligned && i < n; ++s, ++i) {
    *s = rot13_opt1(*s);
  }

  auto transform16 = [&](__m128i v) -> __m128i {
    __m128i lower_case = _mm_or_si128(v, msk_20);
    __m128i gt_a = _mm_cmpgt_epi8(msk_a, lower_case);
    gt_a = _mm_xor_si128(gt_a, msk_ff);
    __m128i le_z = _mm_cmpgt_epi8(msk_z, lower_case);

    __m128i is_alpha = _mm_and_si128(gt_a, le_z);
    __m128i lower_alphas = _mm_and_si128(is_alpha, lower_case);
    __m128i gt_m = _mm_cmpgt_epi8(msk_m, lower_alphas);
    gt_m = _mm_xor_si128(gt_m, msk_ff);
    gt_m = _mm_and_si128(gt_m, is_alpha);

    __m128i off_1 = _mm_and_si128(msk_13, is_alpha);
    __m128i off_2 = _mm_and_si128(msk_26, gt_m);

    v = _mm_add_epi8(v, off_1);
    v = _mm_sub_epi8(v, off_2);
    return v;
  };

  // sse while possible
  for (; i + 16 <= n; s += 16, i += 16) {
    __m128i orig = _mm_load_si128((__m128i *)s);
    orig = transform16(orig);
    _mm_store_si128((__m128i *)s, orig);
  }

  for (; *s && i < n; ++s, ++i) {
    *s = rot13_opt1(*s);
  }
}

void rot13_sse_parallel(char *str, size_t n) {
  size_t cpu_n = get_cpu_count();
  size_t buff_len = n;
  size_t sse_bs = buff_len / cpu_n;
  while (sse_bs % 16) {
    --sse_bs;
  }

  std::vector<size_t> lst_bs(cpu_n, sse_bs);
  const size_t sse_remind = buff_len - (sse_bs * cpu_n);
  lst_bs.back() += sse_remind;

  std::vector<size_t> lst_bo(cpu_n, 0);
  for (size_t i = 1; i < lst_bs.size(); ++i) {
    lst_bo[i] = lst_bo[i - 1] + lst_bs[i - 1];
  }

  run_in_parallel([str, &lst_bo, &lst_bs](size_t i) {
    rot13_sse(str + lst_bo[i], lst_bs[i]);
  });
}
//////////////////////////////////////////////////////////////

void rot13_sse_prefetch(char *str, size_t n) {
  const __m128i msk_20 = _mm_set1_epi8(0x20);
  const __m128i msk_a = _mm_set1_epi8('a');
  const __m128i msk_m = _mm_set1_epi8('m' + 1);
  const __m128i msk_z = _mm_set1_epi8('z' + 1);
  const __m128i msk_13 = _mm_set1_epi8(13);
  const __m128i msk_26 = _mm_set1_epi8(26);
  const __m128i msk_00 = _mm_setzero_si128();
  const __m128i msk_ff = _mm_cmpeq_epi8(msk_00, msk_00);

  uintptr_t p_str = (uintptr_t)str;
  uintptr_t p_aligned = (p_str + 15) & ~((uintptr_t)15);

  // for analigned data we use naive approach
  char *s = str;
  size_t i = 0;
  for (; s != (char *)p_aligned && i < n; ++s, ++i) {
    *s = rot13_opt1(*s);
  }

  const size_t CACHELINE = 64;
  const size_t PF_L1_DIST = 4 * CACHELINE;  // prefetch ~256B ahead to L1
  const size_t PF_L2_DIST = 12 * CACHELINE; // prefetch ~768B ahead to L2

  auto transform16 = [&](__m128i v) -> __m128i {
    __m128i lower_case = _mm_or_si128(v, msk_20);
    __m128i gt_a = _mm_cmpgt_epi8(msk_a, lower_case);
    gt_a = _mm_xor_si128(gt_a, msk_ff);
    __m128i le_z = _mm_cmpgt_epi8(msk_z, lower_case);

    __m128i is_alpha = _mm_and_si128(gt_a, le_z);
    __m128i lower_alphas = _mm_and_si128(is_alpha, lower_case);
    __m128i gt_m = _mm_cmpgt_epi8(msk_m, lower_alphas);
    gt_m = _mm_xor_si128(gt_m, msk_ff);
    gt_m = _mm_and_si128(gt_m, is_alpha);

    __m128i off_1 = _mm_and_si128(msk_13, is_alpha);
    __m128i off_2 = _mm_and_si128(msk_26, gt_m);

    v = _mm_add_epi8(v, off_1);
    v = _mm_sub_epi8(v, off_2);
    return v;
  };

  // cacheline while possible
  for (; i + CACHELINE <= n; s += CACHELINE, i += CACHELINE) {
    if (i + PF_L1_DIST < n)
      _mm_prefetch((const char *)(str + i + PF_L1_DIST), _MM_HINT_T0);
    if (i + PF_L2_DIST < n)
      _mm_prefetch((const char *)(str + i + PF_L2_DIST), _MM_HINT_T1);

    __m128i v0 = _mm_load_si128((__m128i *)(s + 0));
    __m128i v1 = _mm_load_si128((__m128i *)(s + 16));
    __m128i v2 = _mm_load_si128((__m128i *)(s + 32));
    __m128i v3 = _mm_load_si128((__m128i *)(s + 48));

    v0 = transform16(v0);
    v1 = transform16(v1);
    v2 = transform16(v2);
    v3 = transform16(v3);

    _mm_store_si128((__m128i *)(s + 0), v0);
    _mm_store_si128((__m128i *)(s + 16), v1);
    _mm_store_si128((__m128i *)(s + 32), v2);
    _mm_store_si128((__m128i *)(s + 48), v3);
  }

  // sse while possible
  for (; i + 16 <= n; s += 16, i += 16) {
    __m128i orig = _mm_load_si128((__m128i *)s);
    orig = transform16(orig);
    _mm_store_si128((__m128i *)s, orig);
  }

  for (; *s && i < n; ++s, ++i) {
    *s = rot13_opt1(*s);
  }
}

void rot13_sse_prefetch_parallel(char *str, size_t n) {
  size_t cpu_n = get_cpu_count();
  size_t buff_len = n;
  size_t sse_bs = buff_len / cpu_n;
  while (sse_bs % 16) {
    --sse_bs;
  }

  std::vector<size_t> lst_bs(cpu_n, sse_bs);
  const size_t sse_remind = buff_len - (sse_bs * cpu_n);
  lst_bs.back() += sse_remind;

  std::vector<size_t> lst_bo(cpu_n, 0);
  for (size_t i = 1; i < lst_bs.size(); ++i) {
    lst_bo[i] = lst_bo[i - 1] + lst_bs[i - 1];
  }

  run_in_parallel([str, &lst_bo, &lst_bs](size_t i) {
    rot13_sse_prefetch(str + lst_bo[i], lst_bs[i]);
  });
}
//////////////////////////////////////////////////////////////

void rot13_avx2(char *str, size_t n) {
  const __m256i msk_20 = _mm256_set1_epi8(0x20);
  const __m256i msk_a = _mm256_set1_epi8('a');
  const __m256i msk_m = _mm256_set1_epi8('m' + 1);
  const __m256i msk_z = _mm256_set1_epi8('z' + 1);
  const __m256i msk_13 = _mm256_set1_epi8(13);
  const __m256i msk_26 = _mm256_set1_epi8(26);
  const __m256i msk_00 = _mm256_setzero_si256();
  const __m256i msk_ff = _mm256_cmpeq_epi8(msk_00, msk_00);
  uintptr_t p_str = (uintptr_t)str;
  uintptr_t p_aligned = (p_str + 31) & ~((uintptr_t)31);
  char *s = str;
  size_t i = 0;
  for (; s != (char *)p_aligned && i < n; ++s, ++i) {
    *s = rot13_opt1(*s);
  }
  for (; i + 32 <= n; s += 32, i += 32) {
    __m256i orig = _mm256_load_si256((__m256i *)s);
    __m256i lower_case = _mm256_or_si256(orig, msk_20);
    __m256i gt_a = _mm256_cmpgt_epi8(msk_a, lower_case);
    gt_a = _mm256_xor_si256(gt_a, msk_ff);
    __m256i le_z = _mm256_cmpgt_epi8(msk_z, lower_case);
    __m256i is_alpha = _mm256_and_si256(gt_a, le_z);
    __m256i lower_alphas = _mm256_and_si256(is_alpha, lower_case);
    __m256i gt_m = _mm256_cmpgt_epi8(msk_m, lower_alphas);
    gt_m = _mm256_xor_si256(gt_m, msk_ff);
    gt_m = _mm256_and_si256(gt_m, is_alpha);
    __m256i off_1 = _mm256_and_si256(msk_13, is_alpha);
    __m256i off_2 = _mm256_and_si256(msk_26, gt_m);
    orig = _mm256_add_epi8(orig, off_1);
    orig = _mm256_sub_epi8(orig, off_2);
    _mm256_store_si256((__m256i *)s, orig);
  }
  for (; i < n; ++s, ++i) {
    *s = rot13_opt1(*s);
  }
}

void rot13_avx2_parallel(char *str, size_t n) {
  size_t cpu_n = get_cpu_count();
  size_t buff_len = n;
  size_t avx_bs = buff_len / cpu_n;
  while (avx_bs % 32) {
    --avx_bs;
  }

  std::vector<size_t> lst_bs(cpu_n, avx_bs);
  const size_t avx_remind = buff_len - (avx_bs * cpu_n);
  lst_bs.back() += avx_remind;

  std::vector<size_t> lst_bo(cpu_n, 0);
  for (size_t i = 1; i < lst_bs.size(); ++i) {
    lst_bo[i] = lst_bo[i - 1] + lst_bs[i - 1];
  }
  run_in_parallel([str, &lst_bo, &lst_bs](size_t i) {
    rot13_avx2(str + lst_bo[i], lst_bs[i]);
  });
}
//////////////////////////////////////////////////////////////

#if (__AVX512_ENABLED)
void rot13_avx512(char *str, size_t n) {
  assert(str);
  assert(n);
  const __m512i msk_20 = _mm512_set1_epi8(0x20);
  const __m512i msk_a = _mm512_set1_epi8('a');
  const __m512i msk_z = _mm512_set1_epi8('z');
  const __m512i msk_n = _mm512_set1_epi8('n');
  const __m512i msk_13 = _mm512_set1_epi8(13);
  const __m512i msk_26 = _mm512_set1_epi8(26);

  uintptr_t p_str = (uintptr_t)str;
  uintptr_t p_aligned = (p_str + 63) & ~((uintptr_t)63);

  char *s = str;
  size_t i = 0;

  for (; *s && s != (char *)p_aligned && i < n; ++s, ++i) {
    *s = rot13(*s);
  }

  for (; i + 64 <= n; s += 64, i += 64) {
    __m512i orig = _mm512_load_si512((__m512i const *)s);
    __m512i lower_case = _mm512_or_si512(orig, msk_20);

    __mmask64 ge_a = _mm512_cmp_epi8_mask(lower_case, msk_a, _MM_CMPINT_GE);
    __mmask64 le_z = _mm512_cmp_epi8_mask(lower_case, msk_z, _MM_CMPINT_LE);
    __mmask64 is_alpha = ge_a & le_z;

    __mmask64 ge_n = _mm512_cmp_epi8_mask(lower_case, msk_n, _MM_CMPINT_GE);
    __mmask64 wrap_mask = is_alpha & ge_n;

    __m512i tmp = _mm512_mask_add_epi8(orig, is_alpha, orig, msk_13);
    __m512i res = _mm512_mask_sub_epi8(tmp, wrap_mask, tmp, msk_26);
    _mm512_store_si512((__m512i *)s, res);
  }

  for (; *s && i < n; ++s, ++i) {
    *s = rot13(*s);
  }
}
#endif
//////////////////////////////////////////////////////////////

// AUX.h
size_t get_cpu_count() {
  int nprocs = get_nprocs(); // online (available) CPUs
  if (nprocs > 0) {
    return static_cast<size_t>(nprocs);
  }

  // fallback
  unsigned int hc = std::thread::hardware_concurrency();
  return hc > 0 ? hc : 1;
}
//////////////////////////////////////////////////////////////

void run_in_parallel(std::function<void(size_t)> pf_func) {
  size_t cpu_n = get_cpu_count();
  std::vector<std::thread> threads(cpu_n);
  for (size_t i = 0; i < cpu_n; ++i) {
    threads[i] = std::thread([pf_func, i]() { pf_func(i); });
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(i, &cpuset);
    int rc = pthread_setaffinity_np(threads[i].native_handle(),
                                    sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
      std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
  }

  for (auto &t : threads) {
    if (!t.joinable())
      continue;
    t.join();
  }
}
//////////////////////////////////////////////////////////////
