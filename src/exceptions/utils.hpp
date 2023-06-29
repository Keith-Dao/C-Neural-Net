#pragma once
#include <exception>
#include <filesystem>
#include <string>

namespace exceptions::utils {
#pragma region One hot encode
namespace one_hot_encode {
class InvalidLabelIndexException : public std::exception {
  virtual const char *what() const throw();
};
} // namespace one_hot_encode
#pragma endregion One hot encode

#pragma region Image
namespace image {
class InvalidImageFileException : public std::exception {
  std::string file;
  virtual const char *what() const throw();

public:
  InvalidImageFileException(const std::filesystem::path &file) : file(file){};
};
} // namespace image
#pragma endregion Image

#pragma region Normalise
namespace normalise {
class InvalidRangeException : public std::exception {
  virtual const char *what() const throw();
};
} // namespace normalise
#pragma endregion Normalise
} // namespace exceptions::utils