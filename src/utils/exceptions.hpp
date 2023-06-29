#pragma once
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace exceptions {
#pragma region Utils
namespace utils {
#pragma region One hot encode
namespace one_hot_encode {
class InvalidLabelIndexException : public std::exception {
  virtual const char *what() const throw() {
    return "Received a label index greater than the number of classes.";
  }
};
} // namespace one_hot_encode
#pragma endregion One hot encode

#pragma region Image
namespace image {
class InvalidImageFileException : public std::exception {
  std::string file;

  virtual const char *what() const throw() {
    std::string s =
        "The file at " + this->file + " could not be opened as an image.";
    char *result = new char[s.length() + 1];
    std::strcpy(result, s.c_str());
    return result;
  }

public:
  InvalidImageFileException(const std::filesystem::path &file) : file(file){};
};
} // namespace image
#pragma endregion Image

#pragma region Normalise
namespace normalise {
class InvalidRangeException : public std::exception {
  virtual const char *what() const throw() {
    return "An invalid range was provided.";
  }
};
} // namespace normalise
#pragma endregion Normalise
} // namespace utils
#pragma endregion Utils
} // namespace exceptions