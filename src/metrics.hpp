#pragma once
#include <Eigen/Dense>
#include <unordered_set>

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

#pragma region Metrics
const std::unordered_set<std::string> SINGLE_VALUE_METRICS{"accuracy", "loss"};

/*
  The accuracy for the given confusion matrix.
*/
float accuracy(const Eigen::MatrixXi &confusionMatrix);

/*
  The precision for all the classes in the given confusion matrix.
*/
std::vector<float> precision(const Eigen::MatrixXi &confusionMatrix);

/*
  The recall for all the classes in the given confusion matrix.
*/
std::vector<float> recall(const Eigen::MatrixXi &confusionMatrix);

/*
  The f1 score for all the classes in the confusion matrix.
*/
std::vector<float> f1Score(const Eigen::MatrixXi &confusionMatrix);
#pragma endregion Metrics
} // namespace metrics
