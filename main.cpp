#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <math.h>
#include <vector>
#include <set>

#include "queens.h"

std::vector<std::string> split_lines(const std::string &str,
                                     const char separator = '\n');

std::string join_lines(const std::vector<std::string> &lst,
                       const char separator = '\n');

int32_t main(int32_t, char **) {
  std::srand(std::time(nullptr)); // use current time as seed for random generator
  chess_board g(664, 378, 396);
  std::cout << std::boolalpha << g.find_solution(300) << std::endl;
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
