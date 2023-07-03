#pragma once
#include <Eigen/Dense>
#include <filesystem>
#include <functional>
#include <iterator>
#include <memory>
#include <stddef.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

  DatasetIterator(const Batcher *batcher, int i) : batcher(batcher), i(i) {
    if (i != batcher->size()) {
      this->value = (*this->batcher)[this->i];
    }
  }

  reference operator*() const { return this->value; }
  pointer operator->() { return &this->value; }

  // Prefix increment
  DatasetIterator &operator++() {
    if (++this->i < this->batcher->size()) {
      this->value = (*this->batcher)[this->i];
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
  const Batcher *batcher;
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

  DatasetBatcher(){};
  DatasetBatcher(std::filesystem::path root,
                 std::vector<std::filesystem::path> data,
                 preprocessingFunctions preprocessing,
                 std::unordered_map<std::string, int> classesToNum,
                 int batchSize, const KeywordArgs &kwargs);

  DatasetBatcher(std::filesystem::path root,
                 std::vector<std::filesystem::path> data,
                 preprocessingFunctions preprocessing,
                 std::unordered_map<std::string, int> classesToNum,
                 int batchSize);

#pragma region Iterators
  /*
    Iterator at the start of the batches.
  */
  Iterator begin() const;

  /*
    Iterator at the end of the batches.
  */
  Iterator end() const;
#pragma endregion Iterators

#pragma region Properties
#pragma region Size
  /*
    The number of batches.
  */
  virtual int size() const;
#pragma endregion Size
#pragma endregion Properties

#pragma region Builtins
  /*
    Get processed data and labels at batch i (0-based).
  */
  virtual minibatch operator[](int i) const;
#pragma endregion Builtins
};
#pragma endregion Dataset batcher

#pragma region Image loader
class ImageLoader {
  std::filesystem::path root;
  std::vector<std::filesystem::path> trainFiles, testFiles;
  preprocessingFunctions preprocessing;
  std::unordered_map<std::string, int> classesToNum;

protected:
  std::vector<std::string> classes;

public:
  static const preprocessingFunctions standardPreprocessing;

  ImageLoader(){};
  ImageLoader(const std::string &folderPath,
              preprocessingFunctions preprocessing,
              const std::vector<std::string> &fileFormats,
              float trainTestSplit = 1, bool shuffle = true);

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
  virtual std::shared_ptr<DatasetBatcher>
  getBatcher(std::string dataset, int batchSize,
             const DatasetBatcher::KeywordArgs &kwargs =
                 DatasetBatcher::KeywordArgs()) const;
#pragma endregion Batcher

#pragma region Builtins
  /*
    Get the dataset batcher for the selected dataset.
  */
  std::shared_ptr<DatasetBatcher>
  operator()(std::string dataset, int batchSize,
             const DatasetBatcher::KeywordArgs &kwargs =
                 DatasetBatcher::KeywordArgs()) const;
#pragma endregion Builtins
};
#pragma endregion Image loader
} // namespace loader