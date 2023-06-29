#pragma once
#include <exception>
#include <string>

namespace exceptions::model {
class InvalidTotalEpochException : public std::exception {
  int totalEpochs;
  virtual const char *what() const throw();

public:
  InvalidTotalEpochException(int totalEpochs) : totalEpochs(totalEpochs){};
};

class InvalidMetricException : public std::exception {
  std::string metric;
  virtual const char *what() const throw();

public:
  InvalidMetricException(std::string metric) : metric(metric){};
};

class EmptyLayersVectorException : public std::exception {
  virtual const char *what() const throw();
};

class MissingClassesException : public std::exception {
  virtual const char *what() const throw();
};

class ClassHistoryMismatchException : public std::exception {
  int classSize, historySize;
  std::string metric;
  virtual const char *what() const throw();

public:
  ClassHistoryMismatchException(int classSize, int historySize,
                                const std::string &metric)
      : classSize(classSize), historySize(historySize), metric(metric){};
};

class InvalidExtensionException : public std::exception {
  std::string extension;
  virtual const char *what() const throw();

public:
  InvalidExtensionException(const std::string &extension)
      : extension(extension){};
};

class InvalidPlottingMetricException : public std::exception {
  std::string metric;
  virtual const char *what() const throw();

public:
  InvalidPlottingMetricException(const std::string &metric) : metric(metric){};
};
} // namespace exceptions::model