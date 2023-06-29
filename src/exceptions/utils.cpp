#include "utils.hpp"
#include <cstring>

using namespace exceptions::utils;

#pragma region One hot encode
#pragma region InvalidLabelIndexException
const char *one_hot_encode::InvalidLabelIndexException::what() const throw() {
  return "Received a label index greater than the number of classes.";
}
#pragma endregion InvalidLabelIndexException
#pragma endregion One hot encode

#pragma region Image
#pragma region InvalidImageFileException
const char *image::InvalidImageFileException::what() const throw() {
  std::string s =
      "The file at " + this->file + " could not be opened as an image.";
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}
#pragma endregion InvalidImageFileException
#pragma endregion Image

#pragma region Normalise
#pragma region InvalidRangeException
const char *normalise::InvalidRangeException::what() const throw() {
  return "An invalid range was provided.";
}
#pragma endregion InvalidRangeException
#pragma endregion Normalise