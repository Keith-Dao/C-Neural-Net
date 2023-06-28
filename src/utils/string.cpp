#include "string.hpp"
#include <cctype>

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

std::string utils::string::join(const std::vector<std::string> &words,
                                const std::string &joiner) {
  if (words.empty()) {
    return "";
  }
  std::string result;
  for (int i = 0; i < words.size() - 1; ++i) {
    result += words[i] + joiner;
  }
  result += words.back();
  return result;
}
std::string utils::string::floatToString(float num, int precision) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(precision) << num;
  return stream.str();
}

std::string utils::string::capitalise(std::string word) {
  word[0] = std::toupper(word[0]);
  return word;
}

std::string
utils::string::joinWithDifferentLast(std::vector<std::string> words,
                                     const std::string &connector,
                                     const std::string &lastConnector) {
  if (words.empty()) {
    return "";
  }

  std::string last = words.back();
  words.pop_back();
  if (words.empty()) {
    return last;
  }
  return utils::string::join(words, connector) + lastConnector + last;
}