#pragma once
#include <string>
#include <vector>

namespace utils::string {
/*
  Splits the given string by the given delimiter.
*/
std::vector<std::string> split(const std::string &s,
                               const std::string &delimiter);
} // namespace utils::string