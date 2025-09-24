#include "strip_comments.h"

/**
 * Complete the solution so that it strips all text that follows any of a set of
 * comment markers passed in. Any whitespace at the end of the line should also
 * be stripped out.
 */
std::string stripComments(const std::string &str,
                          const std::unordered_set<char> &markers) {
  std::string res;
  res.reserve(str.size());

  bool skipping = false;
  for (auto c : str) {
    if (c == '\n') {
      skipping = false;
      while (!res.empty() && std::isspace(res.back())) {
        res.pop_back();
      }
    }
    
    if (skipping) {
      continue;
    }

    if (markers.find(c) != markers.end()) {
      skipping = true;
      continue;
    }
    res.push_back(c);
  }

  while (!res.empty() && std::isspace(res.back())) {
    res.pop_back();
  }
  return res;
}
