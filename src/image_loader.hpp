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
typedef std::pair<Eigen::MatrixXd, std::vector<int>> minibatch;

#pragma region Dataset batcher
template <typename Batcher> struct DatasetIterator {
  using iterator_category = std::input_iterator_tag;
  using value_type = minibatch;
  using reference = value_type const &;
  using pointer = value_type const *;
  using difference_type = ptrdiff_t;

  DatasetIterator(const Batcher &batcher, int i) : batcher(batcher), i(i) {
    if (i != batcher.size()) {
      this->value = this->batcher[this->i];
    }
  }

  reference operator*() const { return this->value; }
  pointer operator->() { return &this->value; }

  // Prefix increment
  DatasetIterator &operator++() {
    if (++this->i < this->batcher.size()) {
      this->value = this->batcher[this->i];
    }
    return *this;
  }

  // Postfix increment
  DatasetIterator operator++(int) {
    DatasetIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  friend bool operator==(const DatasetIterator &a, const DatasetIterator &b) {
    return a.i == b.i;
  };
  friend bool operator!=(const DatasetIterator &a, const DatasetIterator &b) {
    return a.i != b.i;
  };

private:
  Batcher batcher;
  value_type value;
  int i, end;
};

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

  using Iterator = DatasetIterator<DatasetBatcher>;

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

#pragma region Iterators
  Iterator begin() { return Iterator(*this, 0); }

  Iterator end() { return Iterator(*this, this->size()); }
#pragma endregion Iterators

#pragma region Properties
#pragma region Size
  /*
    The number of batches.
  */
  int size() const;
#pragma endregion Size
#pragma endregion Properties

#pragma region Builtins
  /*
    Get processed data and labels at batch i (0-based).
  */
  minibatch operator[](int i) const;
#pragma endregion Builtins
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