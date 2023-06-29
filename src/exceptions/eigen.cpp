#include "eigen.hpp"

const char *exceptions::eigen::InvalidShapeException::what() const throw() {
  auto pairToString = [](const std::pair<int, int> &pair) {
    return "(" + std::to_string(pair.first) + ", " +
           std::to_string(pair.second) + ")";
  };

  std::string s = "An invalid shape was provided. Expected: " +
                  pairToString(this->expected) +
                  ", got: " + pairToString(this->got) + ".";
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}

const char *exceptions::eigen::EmptyMatrixException::what() const throw() {
  std::string s = this->variable + " cannot be empty.";
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}