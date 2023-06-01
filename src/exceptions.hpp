#pragma once
#include <cstring>
#include <exception>
#include <string>

namespace src_exceptions {
#pragma region Differentiable object
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
#pragma endregion Differentiable object

#pragma region Loss
class InvalidReductionException : public std::exception {
  virtual const char *what() const throw() {
    return "The selected reduction is not valid.";
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
#pragma endregion Loss

#pragma region Utils
class InvalidLabelIndexException : public std::exception {
  virtual const char *what() const throw() {
    return "Received a label index greater than the number of classes.";
  }
};
#pragma endregion Utils

#pragma region Load
class InvalidClassAttributeValue : public std::exception {
  virtual const char *what() const throw() {
    return "Invalid value for class.";
  }
};
#pragma endregion Load

#pragma region Eigen
class InvalidShapeException : public std::exception {
  virtual const char *what() const throw() {
    return "An invalid shape was provided.";
  }
};
#pragma endregion Eigen

#pragma region JSON
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
#pragma endregion JSON
} // namespace src_exceptions