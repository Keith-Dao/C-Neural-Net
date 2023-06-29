#include "differentiable.hpp"

using namespace exceptions::differentiable;

#pragma region BackwardBeforeForwardException
const char *BackwardBeforeForwardException::what() const throw() {
  return "backward was called before forward.";
}
#pragma endregion BackwardBeforeForwardException

#pragma region BackwardCalledInEvalModeException
const char *BackwardCalledInEvalModeException::what() const throw() {
  return "backward cannot be called when set in evaluation mode.";
}
#pragma endregion BackwardCalledInEvalModeException

#pragma region BackwardCalledWithNoInputException
const char *BackwardCalledWithNoInputException::what() const throw() {
  return "backward cannot be called without a stored input.";
}
#pragma endregion BackwardCalledWithNoInputException