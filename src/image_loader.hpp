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

#pragma region Dataset batcher
class DatasetBatcher {
  std::filesystem::path root;
  std::vector<std::filesystem::path> data;
  preprocessingFunctions preprocessing;
  std::unordered_map<std::string, int> classesToNum;
  int batchSize;
  bool dropLast;

public:
  struct KeywordArgs {
    bool shuffle = true, dropLast = false;
  };

  DatasetBatcher(const std::filesystem::path &root,
                 const std::vector<std::filesystem::path> &data,
                 const preprocessingFunctions &preprocessing,
                 const std::unordered_map<std::string, int> &classesToNum,
                 int batchSize, const KeywordArgs kwargs)
      : root(root), data(data), preprocessing(preprocessing),
        classesToNum(classesToNum), batchSize(batchSize),
        dropLast(kwargs.dropLast) {
    if (batchSize < 1) {
      throw exceptions::loader::InvalidBatchSizeException(batchSize);
    }
    if (kwargs.shuffle) {
      std::shuffle(this->data.begin(), this->data.end(),
                   std::default_random_engine{});
    }
  };

  DatasetBatcher(const std::filesystem::path &root,
                 const std::vector<std::filesystem::path> &data,
                 const preprocessingFunctions &preprocessing,
                 const std::unordered_map<std::string, int> &classesToNum,
                 int batchSize)
      : DatasetBatcher(root, data, preprocessing, classesToNum, batchSize,
                       KeywordArgs()){};

  /*
    The number of batches.
  */
  int size() const;

  /*
    Get processed data and labels at batch i (0-based).
  */
  std::pair<Eigen::MatrixXd, std::vector<int>> operator[](int i) const;
};
#pragma endregion Dataset batcher

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

#pragma region Properties
#pragma region Classes
  /*
    Get a copy of the classes in the loader.
  */
  std::vector<std::string> getClasses() const;
#pragma endregion Classes

#pragma region Train files
  /*
    Get a copy of the train files.
  */
  std::vector<std::filesystem::path> getTrainFiles() const;
#pragma endregion Train files

#pragma region Test files
  /*
    Get a copy of the test files.
  */
  std::vector<std::filesystem::path> getTestFiles() const;
#pragma endregion Test files
#pragma endregion Properties

#pragma region Batcher
  /*
    Get the dataset batcher for the selected dataset.
  */
  DatasetBatcher getBatcher(
      std::string dataset, int batchSize,
      const DatasetBatcher::KeywordArgs kwargs = DatasetBatcher::KeywordArgs());
#pragma endregion Batcher

#pragma region Builtins
  /*
    Get the dataset batcher for the selected dataset.
  */
  DatasetBatcher operator()(
      std::string dataset, int batchSize,
      const DatasetBatcher::KeywordArgs kwargs = DatasetBatcher::KeywordArgs());
#pragma endregion Builtins
};
#pragma endregion Image loader
} // namespace loader