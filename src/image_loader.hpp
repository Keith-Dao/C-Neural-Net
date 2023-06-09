#pragma once
#include "exceptions.hpp"
#include "utils.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <filesystem>
#include <random>

namespace loader {
typedef std::vector<std::function<Eigen::MatrixXd(Eigen::MatrixXd)>>
    preprocessingFunctions;

#pragma region Image loader
class ImageLoader {
  std::filesystem::path root;
  std::vector<std::filesystem::path> trainFiles, testFiles;
  preprocessingFunctions preprocessing;
  std::vector<std::string> classes;
  std::unordered_map<std::string, int> classesToNum;

public:
  static const preprocessingFunctions standardPreprocessing;

  ImageLoader(const std::string &folderPath,
              const preprocessingFunctions &preprocessing,
              std::vector<std::string> fileFormats, float trainTestSplit = 1,
              bool shuffle = true)
      : preprocessing(preprocessing) {
    if (trainTestSplit < 0 || trainTestSplit > 1) {
      throw exceptions::loader::InvalidTrainTestSplitException(trainTestSplit);
    }

    this->root = std::filesystem::path(folderPath);
    std::vector<std::filesystem::path> files =
        utils::glob(this->root, fileFormats);
    if (files.empty()) {
      throw exceptions::loader::NoFilesFoundException(this->root, fileFormats);
    }
    if (shuffle) {
      std::shuffle(files.begin(), files.end(), std::default_random_engine{});
    }
    int trainSize = files.size() * trainTestSplit;
    this->trainFiles = std::vector<std::filesystem::path>(
        files.begin(), files.begin() + trainSize);
    this->testFiles = std::vector<std::filesystem::path>(
        files.begin() + trainSize, files.end());

    for (auto entry : std::filesystem::directory_iterator{this->root}) {
      if (entry.is_directory()) {
        this->classes.push_back(entry.path().stem());
      }
    }
    std::sort(this->classes.begin(), this->classes.end());
    for (int i = 0; i < this->classes.size(); ++i) {
      this->classesToNum[this->classes[i]] = i;
    }
  }
};
#pragma endregion Image loader
} // namespace loader