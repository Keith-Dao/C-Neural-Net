#include "model.hpp"
#include <cstring>

using namespace exceptions::model;

#pragma region InvalidTotalEpochException
const char *InvalidTotalEpochException::what() const throw() {
  std::string s =
      "Total epochs must be 0 or greater. Got: " + std::to_string(totalEpochs) +
      ".";
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}
#pragma endregion InvalidTotalEpochException

#pragma region InvalidMetricException
const char *InvalidMetricException::what() const throw() {
  std::string s = metric + " is not a valid metric.";
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}
#pragma region InvalidMetricException

#pragma region EmptyLayersVectorException
const char *EmptyLayersVectorException::what() const throw() {
  return "Layers vector cannot be empty.";
}
#pragma endregion EmptyLayersVectorException

#pragma region MissingClassesException
const char *MissingClassesException::what() const throw() {
  return "Model is missing the classes.";
}
#pragma endregion MissingClassesException

#pragma region ClassHistoryMismatchException
const char *ClassHistoryMismatchException::what() const throw() {
  std::string s = "The number of classes (" + std::to_string(this->classSize) +
                  ") does not match the number of elements for \"" +
                  this->metric + "\" (" + std::to_string(this->historySize) +
                  ").";
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}

#pragma endregion ClassHistoryMismatchException

#pragma region InvalidExtensionException
const char *InvalidExtensionException::what() const throw() {
  std::string s = "File format \"" + this->extension + "\" is not supported.";
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}
#pragma endregion InvalidExtensionException

#pragma region InvalidPlottingMetricException
const char *InvalidPlottingMetricException::what() const throw() {
  std::string s = "Plotting \"" + this->metric + "\" is not supported.";
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}
#pragma endregion InvalidPlottingMetricException