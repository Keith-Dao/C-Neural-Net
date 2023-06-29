#pragma once
#include <exception>

namespace exceptions::load {
class InvalidClassAttributeValue : public std::exception {
  virtual const char *what() const throw();
};
} // namespace exceptions::load