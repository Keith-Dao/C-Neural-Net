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