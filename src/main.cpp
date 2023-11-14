#include <gtest/gtest.h>

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
#ifdef _UNIT_TESTS_
  return main_tests(argc, argv);
#else
  (void)argc;
  (void)argv;
  return 0;

#endif
}
//////////////////////////////////////////////////////////////
