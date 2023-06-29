#include "differentiable.hpp"

const char *
exceptions::differentiable::BackwardBeforeForwardException::what() const
    throw() {
  return "backward was called before forward.";
}

const char *
exceptions::differentiable::BackwardCalledInEvalModeException::what() const
    throw() {
  return "backward cannot be called when set in evaluation mode.";
}

const char *
exceptions::differentiable::BackwardCalledWithNoInputException::what() const
    throw() {
  return "backward cannot be called without a stored input.";
}