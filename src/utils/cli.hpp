#pragma once
#include <string>

namespace utils::cli {
/*
  Ask for a yes/no response. Repeating till a valid response is received and
  returns whether or not the response was yes.
*/
bool getIsYesResponse(const std::string &initialMessage);
} // namespace utils::cli