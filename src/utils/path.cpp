#include "path.hpp"
#include <unordered_set>

std::vector<std::filesystem::path>
utils::path::glob(const std::filesystem::path &path,
                  std::vector<std::string> extensions) {
  std::unordered_set<std::string> extensionSet(extensions.begin(),
                                               extensions.end());
  std::vector<std::filesystem::path> result;
  for (auto const &entry :
       std::filesystem::recursive_directory_iterator(path)) {
    if (!std::filesystem::is_directory(entry.path()) &&
        extensionSet.contains(entry.path().extension())) {
      result.push_back(entry.path());
    }
  };
  return result;
};