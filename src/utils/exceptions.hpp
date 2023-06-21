#pragma once
#include <cstring>
#include <filesystem>
#include <vector>

namespace exceptions {

#pragma region Activation functions
namespace activation {
class InvalidActivationException : public std::exception {
  std::string activation;

  virtual const char *what() const throw() {
    std::string s = this->activation + " is not a valid activation function.";
    char *result = new char[s.length() + 1];
    std::strcpy(result, s.c_str());
    return result;
  }

public:
  InvalidActivationException(std::string activation) : activation(activation){};
};
} // namespace activation
#pragma endregion Activation functions

#pragma region Differentiable
namespace differentiable {
class BackwardBeforeForwardException : public std::exception {
  virtual const char *what() const throw() {
    return "backward was called before forward.";
  }
};

class BackwardCalledInEvalModeException : public std::exception {
  virtual const char *what() const throw() {
    return "backward cannot be called when set in evaluation mode.";
  }
};

class BackwardCalledWithNoInputException : public std::exception {
  virtual const char *what() const throw() {
    return "backward cannot be called without a stored input.";
  }
};
} // namespace differentiable
#pragma endregion Differentiable

#pragma region Eigen
namespace eigen {
class InvalidShapeException : public std::exception {
  virtual const char *what() const throw() {
    return "An invalid shape was provided.";
  }
};

class EmptyMatrixException : public std::exception {
  std::string variable;

  virtual const char *what() const throw() {
    std::string s = this->variable + " cannot be empty.";
    char *result = new char[s.length() + 1];
    std::strcpy(result, s.c_str());
    return result;
  }

public:
  EmptyMatrixException(std::string variable) : variable(variable){};
};
} // namespace eigen
#pragma endregion Eigen

#pragma region Image loader
namespace loader {
class InvalidTrainTestSplitException : public std::exception {
  float split;

  virtual const char *what() const throw() {
    std::string s = "The train test split must be in the [0, 1], got " +
                    std::to_string(split);
    char *result = new char[s.length() + 1];
    std::strcpy(result, s.c_str());
    return result;
  }

public:
  InvalidTrainTestSplitException(float split) : split(split){};
};

class InvalidDatasetException : public std::exception {
  std::string dataset;

  virtual const char *what() const throw() {
    std::string s =
        "An invalid dataset received. Expected \"train\" or \"test\", got " +
        dataset;
    char *result = new char[s.length() + 1];
    std::strcpy(result, s.c_str());
    return result;
  }

public:
  InvalidDatasetException(std::string dataset) : dataset(dataset){};
};

class InvalidBatchSizeException : public std::exception {
  int batchSize;

  virtual const char *what() const throw() {
    std::string s = "The batch size must be greater than or equal 1, got " +
                    std::to_string(batchSize);
    char *result = new char[s.length() + 1];
    std::strcpy(result, s.c_str());
    return result;
  }

public:
  InvalidBatchSizeException(int batchSize) : batchSize(batchSize){};
};

class NoFilesFoundException : public std::exception {
  std::string root, fileFormats;

  virtual const char *what() const throw() {
    std::string s = "No matching files were found at " + this->root +
                    " with the extensions [" + this->fileFormats + "].";
    char *result = new char[s.length() + 1];
    std::strcpy(result, s.c_str());
    return result;
  }

public:
  NoFilesFoundException(const std::filesystem::path &root,
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
  };
};

class InvalidDataShapeAfterPreprocessingException : public std::exception {
  int rows;

  virtual const char *what() const throw() {
    std::string s = "Data must have 1 row after preprocessing, got " +
                    std::to_string(this->rows);
    char *result = new char[s.length() + 1];
    std::strcpy(result, s.c_str());
    return result;
  }

public:
  InvalidDataShapeAfterPreprocessingException(int rows) : rows(rows){};
};
} // namespace loader
#pragma endregion Image loader

#pragma region JSON
namespace json {
class JSONTypeException : public std::exception {
  virtual const char *what() const throw() {
    return "An unexpected type was provided in the JSON data.";
  }
};

class JSONArray2DException : public std::exception {
  virtual const char *what() const throw() {
    return "JSON data should be in the form of a 2D array.";
  }
};
} // namespace json
#pragma endregion JSON

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