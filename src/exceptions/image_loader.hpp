#pragma once
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace exceptions::loader {
class InvalidTrainTestSplitException : public std::exception {
  float split;
  virtual const char *what() const throw();

public:
  InvalidTrainTestSplitException(float split) : split(split){};
};

class InvalidDatasetException : public std::exception {
  std::string dataset;
  virtual const char *what() const throw();

public:
  InvalidDatasetException(std::string dataset) : dataset(dataset){};
};

class InvalidBatchSizeException : public std::exception {
  int batchSize;
  virtual const char *what() const throw();

public:
  InvalidBatchSizeException(int batchSize) : batchSize(batchSize){};
};

class NoFilesFoundException : public std::exception {
  std::string root, fileFormats;
  virtual const char *what() const throw();

public:
  NoFilesFoundException(const std::filesystem::path &root,
                        const std::vector<std::string> &fileFormats);
};

class InvalidDataShapeAfterPreprocessingException : public std::exception {
  int rows;

  virtual const char *what() const throw();

public:
  InvalidDataShapeAfterPreprocessingException(int rows) : rows(rows){};
};
} // namespace exceptions::loader