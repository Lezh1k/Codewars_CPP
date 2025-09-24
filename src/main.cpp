#include <cpuid.h>
#include <gtest/gtest.h>

#include "strip_comments.h"

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

  std::string input("apples, pears # and bananas\n"
                    "grapes\n"
                    "bananas !apples\n");
  std::unordered_set<char> markers = {'#', '!'};
  std::string output = stripComments(input, markers);

  std::cout << output << "\n";

  return 0;
}
//////////////////////////////////////////////////////////////
