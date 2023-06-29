#pragma once
#include <exception>

namespace exceptions::loss {
class InvalidReductionException : public std::exception {
  virtual const char *what() const throw();
};
} // namespace exceptions::loss