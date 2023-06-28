#pragma once
#include <Eigen/Dense>
#include <functional>
#include <unordered_set>
#include <variant>

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

const std::unordered_set<std::string> SINGLE_VALUE_METRICS{"accuracy", "loss"};
const std::unordered_map<
    std::string,
    std::function<std::variant<float, std::vector<float>>(Eigen::MatrixXi)>>
    METRICS{{"accuracy", accuracy},
            {"precision", precision},
            {"recall", recall},
            {"f1_score", f1Score}};
#pragma endregion Metrics
} // namespace metrics
