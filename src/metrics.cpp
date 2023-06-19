#include "metrics.hpp"
#include "utils/exceptions.hpp"

using namespace metrics;

#pragma region Confusion matrix
Eigen::MatrixXi metrics::getNewConfusionMatrix(int numClasses) {
  if (numClasses < 1) {
    throw exceptions::metrics::InvalidNumberOfClassesException(numClasses);
  }
  return Eigen::MatrixXi::Zero(numClasses, numClasses);
}

void metrics::addToConfusionMatrix(Eigen::MatrixXi &confusionMatrix,
                                   const std::vector<int> &predictions,
                                   const std::vector<int> &actual) {
  if (predictions.size() != actual.size()) {
    throw exceptions::metrics::InvalidDatasetException(predictions.size(),
                                                       actual.size());
  }

  for (int i = 0; i < predictions.size(); ++i) {
    confusionMatrix(predictions[i], actual[i])++;
  }
}
#pragma endregion Confusion matrix

#pragma region Metrics
float metrics::accuracy(const Eigen::MatrixXi &confusionMatrix) {
  return (float)confusionMatrix.diagonal().sum() / confusionMatrix.sum();
}

std::vector<float> metrics::precision(const Eigen::MatrixXi &confusionMatrix) {
  Eigen::MatrixXf correctPredictions = confusionMatrix.diagonal().cast<float>(),
                  predicted = confusionMatrix.rowwise().sum().cast<float>(),
                  result = correctPredictions.binaryExpr(
                      predicted,
                      [](float x, float y) { return y == 0 ? 0 : x / y; });
  return std::vector<float>(result.data(), result.data() + result.size());
}
#pragma endregion Metrics