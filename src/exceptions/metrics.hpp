#pragma once
#include <exception>

namespace exceptions::metrics {
class InvalidNumberOfClassesException : public std::exception {
  int numClasses;
  virtual const char *what() const throw();

public:
  InvalidNumberOfClassesException(int numClasses) : numClasses(numClasses){};
};

class InvalidDatasetException : public std::exception {
  int predictionSize, actualSize;
  virtual const char *what() const throw();

public:
  InvalidDatasetException(int predictionSize, int actualSize)
      : predictionSize(predictionSize), actualSize(actualSize){};
};
} // namespace exceptions::metrics