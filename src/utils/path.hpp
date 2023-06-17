#pragma once
#include <filesystem>
#include <vector>

namespace utils::path {
/*
  Recursively find all files with matching extensions
*/
std::vector<std::filesystem::path> glob(const std::filesystem::path &path,
                                        std::vector<std::string> extensions);
} // namespace utils::path
