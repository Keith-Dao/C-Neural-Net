#include "eigen.hpp"
#include <cstring>

using namespace exceptions::eigen;

#pragma region InvalidShapeException
const char *InvalidShapeException::what() const throw() {
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
#pragma endregion InvalidShapeException

#pragma region EmptyMatrixException
const char *EmptyMatrixException::what() const throw() {
  std::string s = this->variable + " cannot be empty.";
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}
#pragma endregion EmptyMatrixException