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

std::string utils::string::join(const std::vector<std::string> &strings,
                                const std::string &joiner) {
  if (strings.empty()) {
    return "";
  }
  std::string result;
  for (int i = 0; i < strings.size() - 1; ++i) {
    result += strings[i] + joiner;
  }
  result += strings.back();
  return result;
}
std::string utils::string::floatToString(float num, int precision) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(precision) << num;
  return stream.str();
}