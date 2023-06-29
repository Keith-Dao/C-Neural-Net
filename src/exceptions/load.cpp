#include "load.hpp"

using namespace exceptions::load;
#pragma region InvalidClassAttributeValue
const char *InvalidClassAttributeValue::what() const throw() {
  return "Invalid value for class.";
}
#pragma endregion InvalidClassAttributeValue