#include "blocks_number.h"

static uint64_t fast_log10(uint64_t v) {
  if (v >= 1000000000000000000) return 18;
  if (v >= 100000000000000000) return 17;
  if (v >= 10000000000000000) return 16;
  if (v >= 1000000000000000) return 15;
  if (v >= 100000000000000) return 14;
  if (v >= 10000000000000) return 13;
  if (v >= 1000000000000) return 12;
  if (v >= 100000000000) return 11;
  if (v >= 10000000000) return 10;
  if (v >= 1000000000) return 9;
  if (v >= 100000000) return 8;
  if (v >= 10000000) return 7;
  if (v >= 1000000) return 6;
  if (v >= 100000) return 5;
  if (v >= 10000) return 4;
  if (v >= 1000) return 3;
  if (v >= 100) return 2;
  if (v >= 10) return 1;
  return 0;
}
//////////////////////////////////////////////////////////////

static uint64_t powi(uint64_t base, uint64_t exp) {
  uint64_t res = 1;
  while (exp) {
    if (exp & 1)
      res *= base;
    exp >>= 1;
    base *= base;
  }
  return res;
}
//////////////////////////////////////////////////////////////

static uint64_t len_of_seq_n(uint64_t n) {
  uint64_t dc = fast_log10(n) + 1; // digits count
  if (dc == 1)
    return n;
  uint64_t pow = powi(10, dc-1);
  uint64_t nearest_len = len_of_seq_n(pow-1); //todo move to some pre-calculated arr
  return (n-pow+1)*dc + nearest_len;
}
//////////////////////////////////////////////////////////////

static uint64_t len_triangle(uint64_t n) {
  /// ((2*a[n] + d*(m-n)) / 2) * (m-n+1)
  /// n = pow; d = log10(m);
  /// ((2 * a[pow] + log10(n)*(n - pow)) / 2) * (n - pow + 1)
  /// 546
  /// len[100:546] + len[10:99] + len[1:9]
  /// 5987
  /// len[1000:5987] + len[100:999] + len[10:99] + len[1:9]

  uint64_t d = 1;
  uint64_t left = 1;
  uint64_t right = n <= 9 ? n : 9;
  uint64_t len = 0;

  while (left <= n) {
    uint64_t am = len_of_seq_n(left);
    long double sum = 2*am + d*(right-left);
    sum /= 2.0l;
    sum *= right - left + 1;
    len += static_cast<uint64_t>(sum);
    left *= 10;
    right = right * 10 + 9;
    if (right > n)
      right = n;
    ++d;
  }
  return len;
}
//////////////////////////////////////////////////////////////

// Method to return Nth character in concatenated
// decimal string
int get_nth_char(uint64_t i) {
  uint64_t     digits = 1U;
  uint64_t     value  = 1U;
  uint64_t     limit  = 9U;
  if (!i--)
    return 0;

  while (i / limit >= digits) {
    const uint64_t old_limit = limit;
    i -= digits * limit;
    ++digits;
    value *= 10U;
    limit *= 10U;

    if (limit <= old_limit)
      break;
  }

  value += i / digits;
  value /= powi(10, (digits - 1) - (i % digits));
  return value % 10U;
}

int solve(uint64_t ix) {
  uint64_t l, r;
  l = 0;
  r = ix; // should be MUCH bigger than n, so!

  // binary search of minimal triangle number which len is greater than requested index
  while (l != r) {
    uint64_t m = (l + r) >> 1;
    uint64_t m_len = len_triangle(m);
    if (m_len >= ix) {
      r = m;
    } else {
      l = m+1;
    }
  }
  uint64_t n = l;
  ix -= len_triangle(n-1); // we do not need everything before nth part
  return get_nth_char(ix);// - '0';
}
