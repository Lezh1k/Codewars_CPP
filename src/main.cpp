#include <algorithm>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

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

  std::vector<int> arr = {1, 2, 0, 3, 4, 0, 5, 8};

  for (auto it : arr) {
    std::cout << it << ' ';
  }
  std::cout << std::endl;

  std::fill(
      std::remove_if(arr.begin(), arr.end(), [](int it) { return it == 0; }),
      arr.end(), 0);

  for (auto it : arr) {
    std::cout << it << ' ';
  }
  std::cout << std::endl;
  return 0;
}
//////////////////////////////////////////////////////////////
