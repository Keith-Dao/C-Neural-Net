#include "json.hpp"

using namespace exceptions::json;

#pragma region JSONTypeException
const char *JSONTypeException::what() const throw() {
  return "An unexpected type was provided in the JSON data.";
}
#pragma endregion JSONTypeException

#pragma region JSONArray2DException
const char *JSONArray2DException::what() const throw() {
  return "JSON data should be in the form of a 2D array.";
}
#pragma endregion JSONArray2DException