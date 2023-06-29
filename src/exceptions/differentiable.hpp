#pragma once
#include <exception>

namespace exceptions::differentiable {
class BackwardBeforeForwardException : public std::exception {
  virtual const char *what() const throw();
};

class BackwardCalledInEvalModeException : public std::exception {
  virtual const char *what() const throw();
};

class BackwardCalledWithNoInputException : public std::exception {
  virtual const char *what() const throw();
};
} // namespace exceptions::differentiable