#pragma once
#include <exception>

namespace exceptions::json {
class JSONTypeException : public std::exception {
  virtual const char *what() const throw();
};

class JSONArray2DException : public std::exception {
  virtual const char *what() const throw();
};
} // namespace exceptions::json