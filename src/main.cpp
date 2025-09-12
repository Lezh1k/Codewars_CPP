#include "rot13_bench.h"
#include <cpuid.h>
#include <gtest/gtest.h>

#ifdef _UNIT_TESTS_
int main_tests(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
//////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
#ifdef _UNIT_TESTS_
  return main_tests(argc, argv);
#endif
  (void)argc;
  (void)argv;
  return rot13_bench_launch();
}
//////////////////////////////////////////////////////////////
