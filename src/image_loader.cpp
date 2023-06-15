#include "image_loader.hpp"
#include "exceptions.hpp"
#include "utils.hpp"
#include <filesystem>
#include <stdexcept>

using namespace loader;

#pragma region Dataset batcher
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
    Eigen::MatrixXd image = utils::openImageAsMatrix(path);
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
    utils::normaliseImage, utils::flatten};

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