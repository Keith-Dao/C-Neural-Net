#pragma once
#include <iostream>
#include <string>

namespace utils::cli {
/*
Check if the response is yes. If it not either yes or no, ask again.
*/
bool isYes(std::string response);
} // namespace utils::cli