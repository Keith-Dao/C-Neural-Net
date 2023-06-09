#pragma once
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace exceptions {
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

namespace loss {
class InvalidReductionException : public std::exception {
  virtual const char *what() const throw() {
    return "The selected reduction is not valid.";
  }
};
} // namespace loss

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
} // namespace loader

namespace utils {
namespace one_hot_encode {
class InvalidLabelIndexException : public std::exception {
  virtual const char *what() const throw() {
    return "Received a label index greater than the number of classes.";
  }
};
} // namespace one_hot_encode

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

namespace normalise {
class InvalidRangeException : public std::exception {
  virtual const char *what() const throw() {
    return "An invalid range was provided.";
  }
};
} // namespace normalise
} // namespace utils

namespace load {
class InvalidClassAttributeValue : public std::exception {
  virtual const char *what() const throw() {
    return "Invalid value for class.";
  }
};
} // namespace load

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
} // namespace exceptions