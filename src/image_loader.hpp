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
                 int batchSize, const KeywordArgs kwargs);

  DatasetBatcher(const std::filesystem::path &root,
                 const std::vector<std::filesystem::path> &data,
                 const preprocessingFunctions &preprocessing,
                 const std::unordered_map<std::string, int> &classesToNum,
                 int batchSize);

#pragma region Iterators
  /*
    Iterator at the start of the batches.
  */
  Iterator begin();

  /*
    Iterator at the end of the batches.
  */
  Iterator end();
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
              bool shuffle = true);

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