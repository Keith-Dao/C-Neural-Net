#include "activation_functions.hpp"
#include <cstring>

const char *exceptions::activation::InvalidActivationException::what() const
    throw() {
  std::string s = this->activation + " is not a valid activation function.";
  char *result = new char[s.length() + 1];
  std::strcpy(result, s.c_str());
  return result;
}