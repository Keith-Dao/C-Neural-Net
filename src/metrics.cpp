#include "metrics.hpp"
#include "exceptions/metrics.hpp"
#include <Eigen/Dense>
#include <algorithm>

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
  Eigen::VectorXf correctPredictions = confusionMatrix.diagonal().cast<float>(),
                  predicted = confusionMatrix.rowwise().sum().cast<float>(),
                  result = correctPredictions.binaryExpr(
                      predicted,
                      [](float x, float y) { return y == 0 ? 0 : x / y; });
  return std::vector<float>(result.data(), result.data() + result.size());
}

std::vector<float> metrics::recall(const Eigen::MatrixXi &confusionMatrix) {
  Eigen::VectorXf correctPredictions = confusionMatrix.diagonal().cast<float>(),
                  actual = confusionMatrix.colwise().sum().cast<float>(),
                  result = correctPredictions.binaryExpr(
                      actual,
                      [](float x, float y) { return y == 0 ? 0 : x / y; });
  return std::vector<float>(result.data(), result.data() + result.size());
}

std::vector<float> metrics::f1Score(const Eigen::MatrixXi &confusionMatrix) {
  int size = confusionMatrix.rows();
  Eigen::VectorXf precision_ = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(
                      precision(confusionMatrix).data(), size),
                  recall_ = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(
                      recall(confusionMatrix).data(), size),
                  result = (2 * precision_.cwiseProduct(recall_))
                               .binaryExpr((precision_ + recall_),
                                           [](float x, float y) {
                                             return y == 0 ? 0 : x / y;
                                           });
  return std::vector<float>(result.data(), result.data() + result.size());
}
#pragma endregion Metrics