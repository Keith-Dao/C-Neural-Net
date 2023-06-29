#include "metrics.hpp"
#include <cstring>
#include <string>

using namespace exceptions::metrics;

#pragma region InvalidNumberOfClassesException
const char *InvalidNumberOfClassesException::what() const throw() {
  std::string s =
      "The number of classes must be > 0. Got: " + std::to_string(numClasses) +
      ".";
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}
#pragma endregion InvalidNumberOfClassesException

#pragma region InvalidDatasetException
const char *InvalidDatasetException::what() const throw() {
  std::string s = "The length of predictions (" +
                  std::to_string(this->predictionSize) + ") and actual (" +
                  std::to_string(this->actualSize) + ") does not match.";
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}
#pragma endregion InvalidDatasetException