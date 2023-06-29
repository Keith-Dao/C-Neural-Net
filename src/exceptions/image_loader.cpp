#include "image_loader.hpp"
#include <cstring>

using namespace exceptions::loader;

#pragma region InvalidTrainTestSplitException
const char *InvalidTrainTestSplitException::what() const throw() {
  std::string s = "The train test split must be in the [0, 1], got " +
                  std::to_string(split);
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}
#pragma endregion InvalidTrainTestSplitException

#pragma region InvalidDatasetException
const char *InvalidDatasetException::what() const throw() {
  std::string s =
      "An invalid dataset received. Expected \"train\" or \"test\", got " +
      dataset;
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}
#pragma endregion InvalidDatasetException

#pragma region InvalidBatchSizeException
const char *InvalidBatchSizeException::what() const throw() {
  std::string s = "The batch size must be greater than or equal 1, got " +
                  std::to_string(this->batchSize);
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}
#pragma endregion InvalidBatchSizeException

#pragma region NoFilesFoundException
const char *NoFilesFoundException::what() const throw() {
  std::string s = "No matching files were found at " + this->root +
                  " with the extensions [" + this->fileFormats + "].";
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}

NoFilesFoundException::NoFilesFoundException(
    const std::filesystem::path &root,
    const std::vector<std::string> &fileFormats)
    : root(root) {
  if (fileFormats.empty()) {
    this->fileFormats = "";
    return;
  }
  this->fileFormats = fileFormats[0];
  for (int i = 1; i < fileFormats.size(); ++i) {
    this->fileFormats += ", " + fileFormats[i];
  }
}
#pragma endregion NoFilesFoundException

#pragma region InvalidDataShapeAfterPreprocessingException
const char *InvalidDataShapeAfterPreprocessingException::what() const throw() {
  std::string s = "Data must have 1 row after preprocessing, got " +
                  std::to_string(this->rows);
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}
#pragma endregion InvalidDataShapeAfterPreprocessingException