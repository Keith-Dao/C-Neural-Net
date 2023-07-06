#pragma once
#include <string>

namespace utils::cli {
/*
  Ask for a yes/no response. Repeating till a valid response is received and
  returns whether or not the response was yes.
*/
bool getIsYesResponse(const std::string &initialMessage);

/*
  Print the given message to std::cout in the warning colours.
*/
void printWarning(const std::string &message);

/*
  Print the given message to std::cout in the error colours.
*/
void printError(const std::string &message);

/*
  Prompt path from input, while saving any inputs to the input history.
*/
std::string promptPath(const std::string &prompt);
} // namespace utils::cli