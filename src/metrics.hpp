#pragma once
#include <Eigen/Dense>

namespace metrics {
#pragma region Confusion matrix
/*
  Create a new confusion matrix with the given number of classes, where the rows
  the predicted class amd the columns represent the actual class.
*/
Eigen::MatrixXi getNewConfusionMatrix(int numClasses);

/*
  Add the predictions to the given confusion matrix.
*/
void addToConfusionMatrix(Eigen::MatrixXi &confusionMatrix,
                          const std::vector<int> &predictions,
                          const std::vector<int> &actual);
#pragma endregion Confusion matrix
} // namespace metrics
