#include "loss.hpp"

using namespace exceptions::loss;

#pragma region InvalidReductionException
const char *InvalidReductionException::what() const throw() {
  return "The selected reduction is not valid.";
}
#pragma endregion InvalidReductionException