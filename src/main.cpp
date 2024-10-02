#include <gtest/gtest.h>

#include "tip_toe_through_circles.h"

#ifdef _UNIT_TESTS_
int main_tests(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
//////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
  (void)argc;
  (void)argv;
  run_all_test_cases(false);
  return 0;
}
