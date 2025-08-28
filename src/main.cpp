#include "rot13_bench.h"
#include <gtest/gtest.h>

#ifdef _UNIT_TESTS_
int main_tests(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
//////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  return rot13_bench_launch();
}
//////////////////////////////////////////////////////////////
