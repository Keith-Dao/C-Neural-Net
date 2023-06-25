#include "string.hpp"

std::vector<std::string> utils::string::split(const std::string &s,
                                              const std::string &delimiter) {
  std::vector<std::string> result;
  size_t start = 0, end;
  while ((end = s.find(delimiter, start)) != std::string::npos) {
    result.push_back(s.substr(start, end - start));
    start = end + delimiter.size();
  }
  result.push_back(s.substr(start));
  return result;
}