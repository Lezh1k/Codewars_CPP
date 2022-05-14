#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <math.h>
#include <vector>

std::vector<std::string> split_lines(const std::string &str,
                                     const char separator = '\n');

std::string join_lines(const std::vector<std::string> &lst,
                       const char separator = '\n');

#include "game_model.h"
#include "vertex.h"

int32_t main(int32_t, char **) {
  std::vector<std::vector<std::vector<int32_t>>> puzzles = {
    //    {
    //      {22, 1, 10, 9, 4},
    //      {11, 8, 2, 5, 18},
    //      {16, 6, 12, 19, 14},
    //      {7, 23, 21, 15, 24},
    //      {3, 13, 0, 17, 20},
    //    },
    {
      {4, 1, 3},
      {2, 8, 0},
      {7, 6, 5}
    },
//    {
//      {10, 3, 6, 4},
//      {1, 5, 8, 0},
//      {2, 13, 7, 15},
//      {14, 9, 12, 11}
//    },
//    {
//      {3, 7, 14, 15, 10},
//      {1, 0, 5, 9, 4},
//      {16, 2, 11, 12, 8},
//      {17, 6, 13, 18, 20},
//      {21, 22, 23, 19, 24}
//    },
//    {
//      {3, 9, 11, 7},
//      {1, 12, 13, 4},
//      {8, 2, 14, 0},
//      {6, 10, 15, 5}
//    }
  };

  for (auto &puzzle : puzzles) {
    Game g(puzzle);
//    auto start = std::chrono::system_clock::now();
//    std::vector<int32_t> solution2 = g.find_solution_a_star();
//    auto end = std::chrono::system_clock::now();
//    auto dur = end - start;

//    std::cout
//        << "a* took: "
//        << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count()
//        << "ms\n";
//    for (auto it : solution2) {
//      std::cout << it << " ";
//    }
//    std::cout << "\n";

//    start = std::chrono::system_clock::now();
//    std::vector<int32_t> solution = g.find_solution_ida_star();
//    end = std::chrono::system_clock::now();
//    dur = end - start;

//    std::cout
//        << "ida took: "
//        << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count()
//        << "ms\n";
//    for (auto it : solution) {
//      std::cout << it << " ";
//    }
//    std::cout << "\n";

    auto start = std::chrono::system_clock::now();
    std::vector<int32_t> solution3 = g.find_solution_by_strategy();
    auto end = std::chrono::system_clock::now();
    auto dur = end - start;

//    std::cout
//        << "ida took: "
//        << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count()
//        << "ms\n";
//    for (auto it : solution3) {
//      std::cout << it << " ";
//    }
//    std::cout << "\n";
  }

  return 0;

  std::string &&input = join_lines(
      {"+------------+", "|            |", "|            |", "|            |",
       "+------+-----+", "|      |     |", "|      |     |", "+------+-----+"});
  std::cout << "input:\n" << input << "\n\n\n";
  std::vector<std::string> splitted = split_lines(input, '\n');

  std::cout << "splitted:\n";
  for (auto &s : splitted) {
    std::cout << s << "\n";
  }
  std::cout << std::endl;
  return 0;
}
//////////////////////////////////////////////////////////////

std::vector<std::string> split_lines(const std::string &str,
                                     const char separator /*= '\n'*/) {
  std::vector<std::string> res;
  size_t b = 0, e;
  do {
    e = str.find_first_of(separator, b);
    res.push_back(str.substr(b, e - b));
    b = e + 1;
  } while (e != std::string::npos);
  return res;
}
//////////////////////////////////////////////////////////////

std::string join_lines(const std::vector<std::string> &lst,
                       const char separator) {
  size_t cap = 0;
  for (auto &s : lst)
    cap += s.size();
  cap += lst.size();

  std::string res;
  res.reserve(cap);
  for (auto &s : lst) {
    res += s;
    res.push_back(separator);
  }
  res.pop_back(); // last separator
  return res;
}
//////////////////////////////////////////////////////////////
