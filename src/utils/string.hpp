#pragma once
#include <iomanip>
#include <ios>
#include <sstream>
#include <string>
#include <vector>

namespace utils::string {
/*
  Splits the given string by the given delimiter.
*/
std::vector<std::string> split(const std::string &s,
                               const std::string &delimiter);

/*
  Joins the given strings by the given token.
*/
std::string join(const std::vector<std::string> &strings,
                 const std::string &joiner);

/*
  Convert a float to a set precision string.
*/
std::string floatToString(float num, int precision);
} // namespace utils::string