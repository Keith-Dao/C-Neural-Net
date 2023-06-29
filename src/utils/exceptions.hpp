#pragma once
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace exceptions {
#pragma region Load methods
namespace load {
class InvalidClassAttributeValue : public std::exception {
  virtual const char *what() const throw() {
    return "Invalid value for class.";
  }
};
} // namespace load
#pragma endregion Load methods

#pragma region Loss
namespace loss {
class InvalidReductionException : public std::exception {
  virtual const char *what() const throw() {
    return "The selected reduction is not valid.";
  }
};
} // namespace loss
#pragma endregion Loss

#pragma region Metrics
namespace metrics {
class InvalidNumberOfClassesException : public std::exception {
  int numClasses;

  virtual const char *what() const throw() {
    std::string s = "The number of classes must be > 0. Got: " +
                    std::to_string(numClasses) + ".";
    char *result = new char[s.length() + 1];
    std::strcpy(result, s.c_str());
    return result;
  }

public:
  InvalidNumberOfClassesException(int numClasses) : numClasses(numClasses){};
};

class InvalidDatasetException : public std::exception {
  int predictionSize, actualSize;

  virtual const char *what() const throw() {
    std::string s = "The length of predictions (" +
                    std::to_string(predictionSize) + ") and actual (" +
                    std::to_string(actualSize) + ") does not match.";
    char *result = new char[s.length() + 1];
    std::strcpy(result, s.c_str());
    return result;
  }

public:
  InvalidDatasetException(int predictionSize, int actualSize)
      : predictionSize(predictionSize), actualSize(actualSize){};
};
} // namespace metrics
#pragma endregion Metrics

#pragma region Model
namespace model {
class InvalidTotalEpochException : public std::exception {
  int totalEpochs;

  virtual const char *what() const throw() {
    std::string s = "Total epochs must be 0 or greater. Got: " +
                    std::to_string(totalEpochs) + ".";
    char *result = new char[s.length() + 1];
    std::strcpy(result, s.c_str());
    return result;
  }

public:
  InvalidTotalEpochException(int totalEpochs) : totalEpochs(totalEpochs){};
};

class InvalidMetricException : public std::exception {
  std::string metric;

  virtual const char *what() const throw() {
    std::string s = metric + " is not a valid metric.";
    char *result = new char[s.length() + 1];
    std::strcpy(result, s.c_str());
    return result;
  }

public:
  InvalidMetricException(std::string metric) : metric(metric){};
};

class EmptyLayersVectorException : public std::exception {
  virtual const char *what() const throw() {
    return "Layers vector cannot be empty.";
  }
};

class MissingClassesException : public std::exception {
  virtual const char *what() const throw() {
    return "Model is missing the classes.";
  }
};

class ClassHistoryMismatchException : public std::exception {
  int classSize, historySize;
  std::string metric;

  virtual const char *what() const throw() {
    std::string s =
        "The number of classes (" + std::to_string(this->classSize) +
        ") does not match the number of elements for \"" + this->metric +
        "\" (" + std::to_string(this->historySize) + ").";
    char *result = new char[s.length() + 1];
    std::strcpy(result, s.c_str());
    return result;
  }

public:
  ClassHistoryMismatchException(int classSize, int historySize,
                                const std::string &metric)
      : classSize(classSize), historySize(historySize), metric(metric){};
};

class InvalidExtensionException : public std::exception {
  std::string extension;

  virtual const char *what() const throw() {
    std::string s = "File format \"" + this->extension + "\" is not supported.";
    char *result = new char[s.length() + 1];
    std::strcpy(result, s.c_str());
    return result;
  }

public:
  InvalidExtensionException(const std::string &extension)
      : extension(extension){};
};

class InvalidPlottingMetricException : public std::exception {
  std::string metric;

  virtual const char *what() const throw() {
    std::string s = "Plotting \"" + this->metric + "\" is not supported.";
    char *result = new char[s.length() + 1];
    std::strcpy(result, s.c_str());
    return result;
  }

public:
  InvalidPlottingMetricException(const std::string &metric) : metric(metric){};
};
} // namespace model
#pragma endregion Model

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