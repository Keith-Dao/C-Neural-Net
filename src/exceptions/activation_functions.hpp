#pragma once
#include <exception>
#include <string>

namespace exceptions::activation {
class InvalidActivationException : public std::exception {
  std::string activation;

  virtual const char *what() const throw();

public:
  InvalidActivationException(std::string activation) : activation(activation){};
};
} // namespace exceptions::activation