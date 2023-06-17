#include "image_loader.hpp"
#include "utils/exceptions.hpp"
#include "utils/image.hpp"
#include "utils/matrix.hpp"
#include "utils/path.hpp"
#include <random>

using namespace loader;

#pragma region Dataset batcher
#pragma region Constructor
DatasetBatcher::DatasetBatcher(
    const std::filesystem::path &root,
    const std::vector<std::filesystem::path> &data,
    const preprocessingFunctions &preprocessing,
    const std::unordered_map<std::string, int> &classesToNum, int batchSize,
    const KeywordArgs kwargs)
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

DatasetBatcher::DatasetBatcher(
    const std::filesystem::path &root,
    const std::vector<std::filesystem::path> &data,
    const preprocessingFunctions &preprocessing,
    const std::unordered_map<std::string, int> &classesToNum, int batchSize)
    : DatasetBatcher(root, data, preprocessing, classesToNum, batchSize,
                     KeywordArgs()){};
#pragma endregion Constructor

#pragma region Iterators
DatasetBatcher::Iterator DatasetBatcher::begin() { return Iterator(*this, 0); }

DatasetBatcher::Iterator DatasetBatcher::end() {
  return Iterator(*this, this->size());
}
#pragma endregion Iterators

#pragma region Properties
#pragma region Size
int DatasetBatcher::size() const {
  return (this->data.size() + (this->dropLast ? 0 : this->batchSize - 1)) /
         this->batchSize;
}
#pragma endregion Size
#pragma endregion Properties

#pragma region Builtins
minibatch DatasetBatcher::operator[](int batch) const {
  if (batch >= this->size() || batch < 0) {
    throw std::out_of_range("Batch is out of range.");
  }

  Eigen::MatrixXd
      result; // Set dimensions later when the number of columns is known.
  std::vector<int> labels;
  int stopIndex =
      std::min(this->data.size(), (unsigned long)(batch + 1) * this->batchSize);
  for (int i = batch * this->batchSize; i < stopIndex; ++i) {
    std::filesystem::path path = this->data[i];

    // Process image data
    Eigen::MatrixXd image = utils::image::openAsMatrix(path);
    for (auto step : this->preprocessing) {
      image = step(image);
    }
    if (image.rows() != 1) {
      throw exceptions::loader::InvalidDataShapeAfterPreprocessingException(
          image.rows());
    }
    if (result.size() == 0) {
      result = Eigen::MatrixXd(stopIndex - i, image.cols());
    }
    result.row(i - batch * this->batchSize) = image.row(0);

    // Process label
    std::string label = *std::filesystem::relative(path, this->root).begin();
    labels.push_back(this->classesToNum.at(label));
  }

  return std::make_pair(result, labels);
}
#pragma endregion Builtins
#pragma endregion Dataset batcher

#pragma region Image loader
const preprocessingFunctions ImageLoader::standardPreprocessing{
    utils::image::normalise, utils::matrix::flatten};

#pragma region Constructor
ImageLoader::ImageLoader(const std::string &folderPath,
                         const preprocessingFunctions &preprocessing,
                         std::vector<std::string> fileFormats,
                         float trainTestSplit, bool shuffle)
    : preprocessing(preprocessing) {
  if (trainTestSplit < 0 || trainTestSplit > 1) {
    throw exceptions::loader::InvalidTrainTestSplitException(trainTestSplit);
  }

  this->root = std::filesystem::path(folderPath);
  std::vector<std::filesystem::path> files =
      utils::path::glob(this->root, fileFormats);
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

#pragma endregion Constructor

#pragma region Properties
#pragma region Classes
std::vector<std::string> ImageLoader::getClasses() const {
  return this->classes;
};
#pragma endregion Classes

#pragma region Train files
std::vector<std::filesystem::path> ImageLoader::getTrainFiles() const {
  return this->trainFiles;
}
#pragma endregion Train files

#pragma region Test files
std::vector<std::filesystem::path> ImageLoader::getTestFiles() const {
  return this->testFiles;
}
#pragma endregion Test files
#pragma endregion Properties

#pragma region Batcher
DatasetBatcher
ImageLoader::getBatcher(std::string dataset, int batchSize,
                        const DatasetBatcher::KeywordArgs kwargs) {
  if (dataset != "train" && dataset != "test") {
    throw exceptions::loader::InvalidDatasetException(dataset);
  }
  return DatasetBatcher(
      this->root, dataset == "train" ? this->trainFiles : this->testFiles,
      this->preprocessing, this->classesToNum, batchSize, kwargs);
};
#pragma endregion Batcher

#pragma region Builtins
DatasetBatcher
ImageLoader::operator()(std::string dataset, int batchSize,
                        const DatasetBatcher::KeywordArgs kwargs) {
  return this->getBatcher(dataset, batchSize, kwargs);
}
#pragma endregion Builtins
#pragma endregion Image loader