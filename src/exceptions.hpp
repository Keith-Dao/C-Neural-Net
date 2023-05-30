#pragma once
#include <exception>

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