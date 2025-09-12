#include <assert.h>
#include <gtest/gtest.h>

#include "rot13.h"

struct Rot13CharCase {
  const char *name;
  char (*fn)(char);
};
class Rot13CharTest : public ::testing::TestWithParam<Rot13CharCase> {};

TEST_P(Rot13CharTest, matches_expected) {
  const Rot13CharCase &p = GetParam();
  ASSERT_EQ(' ', p.fn(' '));
  ASSERT_EQ('1', p.fn('1'));
  ASSERT_EQ('[', p.fn('['));
  ASSERT_EQ('?', p.fn('?'));
  ASSERT_EQ('!', p.fn('!'));

  for (uint8_t i = 0; i < 'A'; ++i) {
    ASSERT_EQ((char)i, p.fn((char)i));
  }
  ASSERT_NE('a', p.fn('a'));
  ASSERT_EQ('a', p.fn('n'));
  ASSERT_EQ('o', p.fn('b'));
}
//////////////////////////////////////////////////////////////

INSTANTIATE_TEST_SUITE_P(
    rot13_char, Rot13CharTest,
    ::testing::Values(Rot13CharCase{"rot13_naive", rot13_dummy},
                      Rot13CharCase{"rot13_opt1", rot13_opt1}),
    [](const ::testing::TestParamInfo<Rot13CharCase> &info) {
      return std::string(info.param.name);
    });
static std::string orig("123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
                        "abcdefghijklmnopqrstuvwxyz{|}~\t\n\r");
static std::string encoded("123456789:;<=>?@NOPQRSTUVWXYZABCDEFGHIJKLM[\\]^_`"
                           "nopqrstuvwxyzabcdefghijklm{|}~\t\n\r");

static std::unique_ptr<char[]> get_orig_buff(size_t repeats,
                                             size_t &out_buff_len) {
  out_buff_len = orig.size() * repeats + 1;
  std::unique_ptr<char[]> work(new char[out_buff_len]);
  assert(work);
  memset(work.get(), 0, out_buff_len);
  for (size_t offset = 0; offset < out_buff_len - orig.size();
       offset += orig.size()) {
    memcpy(work.get() + offset, orig.c_str(), orig.size());
  }
  return work;
}

static std::string get_expected_str(size_t repeats, size_t &out_buff_len) {
  std::string res;
  out_buff_len = orig.size() * repeats + 1;
  res.reserve(out_buff_len);
  for (size_t i = 0; i < repeats; ++i) {
    res += encoded;
  }
  return res;
}
//////////////////////////////////////////////////////////////

struct Rot13StrCase {
  const char *name;
  void (*fn)(char *, size_t);
  size_t repeats;
};
class Rot13StrTest : public ::testing::TestWithParam<Rot13StrCase> {};

TEST_P(Rot13StrTest, matches_expected) {
  const Rot13StrCase &p = GetParam();
  size_t buff_len = 0;
  std::unique_ptr<char[]> work = get_orig_buff(p.repeats, buff_len);
  std::string expected = get_expected_str(p.repeats, buff_len);
  ASSERT_STRNE(expected.c_str(), work.get());
  p.fn(work.get(), buff_len);
  ASSERT_STREQ(expected.c_str(), work.get());
}
//////////////////////////////////////////////////////////////

INSTANTIATE_TEST_SUITE_P(
    rot13_str, Rot13StrTest,
    ::testing::Values(
        Rot13StrCase{"rot13_naive", rot13_naive, 4},
        Rot13StrCase{"rot13_naive_parallel", rot13_naive_parallel, 40},
        Rot13StrCase{"rot13_opt1", rot13_opt1, 9},
        Rot13StrCase{"rot13_opt1_parallel", rot13_opt1_parallel, 90},
        Rot13StrCase{"rot13_lut", rot13_lut, 5},
        Rot13StrCase{"rot13_lut_parallel", rot13_lut_parallel, 50},
        Rot13StrCase{"rot13_sse", rot13_sse, 20},
        Rot13StrCase{"rot13_sse_parallel", rot13_sse_parallel, 200},
        Rot13StrCase{"rot13_sse_prefetch", rot13_sse_prefetch, 20},
        Rot13StrCase{"rot13_sse_prefetch_parallel", rot13_sse_prefetch_parallel,
                     200},
        Rot13StrCase{"rot13_avx2", rot13_avx2, 50},
        Rot13StrCase{"rot13_avx2_parallel", rot13_avx2_parallel, 500},
        Rot13StrCase{"rot13_cuda", cuda_rot13, 500},
        Rot13StrCase{"rot13_cuda_vect", cuda_rot13_vect, 3000}),
    [](const ::testing::TestParamInfo<Rot13StrCase> &info) {
      return std::string(info.param.name);
    });
