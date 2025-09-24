#ifndef STRIP_COMMENTS_H
#define STRIP_COMMENTS_H

#include <string>
#include <unordered_set>

std::string stripComments(const std::string &str,
                          const std::unordered_set<char> &markers);
#endif
