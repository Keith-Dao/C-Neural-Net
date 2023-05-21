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

#pragma region Shape
class InvalidShapeException : public std::exception {
  virtual const char *what() const throw() {
    return "An invalid shape was provided.";
  }
};
#pragma endregion Shape
} // namespace src_exceptions