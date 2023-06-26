#pragma once
#include "cross_entropy_loss.hpp"
#include "image_loader.hpp"
#include "linear.hpp"
#include <nlohmann/json.hpp>
#include <variant>
#include <vector>

using json = nlohmann::json;

namespace model {
typedef std::vector<std::variant<float, std::vector<float>>> metricHistoryValue;

class Model {
  bool eval = false;
  std::vector<linear::Linear> layers;
  loss::CrossEntropyLoss loss;
  int totalEpochs;
  std::unordered_map<std::string, metricHistoryValue> trainMetrics,
      validationMetrics;
  std::vector<std::string> classes;

public:
  struct KeywordArgs {
    int totalEpochs = 0;
    std::vector<std::string> classes;
    std::unordered_map<std::string, metricHistoryValue> trainMetrics,
        validationMetrics;

    /*
      Set the train metrics from the metric types.
    */
    void setTrainMetricsFromMetricTypes(std::vector<std::string> metrics);
    /*
      Set the validation from the metric types.
    */
    void setValidationMetricsFromMetricTypes(std::vector<std::string> metrics);
  };

  Model(std::vector<linear::Linear> layers, loss::CrossEntropyLoss loss,
        KeywordArgs kwargs);
  Model(std::vector<linear::Linear> layers, loss::CrossEntropyLoss loss);

#pragma region Properties
#pragma region Evaluation mode
  /*
    Get the model's evaluation mode.
  */
  bool getEval() const;
  /*
    Set the model's evaluation mode.
  */
  void setEval(bool eval);
#pragma endregion Evaluation mode

#pragma region Layers
  /*
    Get the model's layers
  */
  std::vector<linear::Linear> getLayers() const;

  /*
    Set the model's layers
  */
  void setLayers(const std::vector<linear::Linear> &layers);
#pragma endregion Layers

#pragma region Loss
  /*
    Get the model's loss.
  */
  loss::CrossEntropyLoss getLoss() const;

  /*
    Set the model's loss.
  */
  void setLoss(const loss::CrossEntropyLoss &loss);
#pragma endregion Loss

#pragma region Total epochs
  /*
    Get the model's total epochs
  */
  int getTotalEpochs() const;

  /*
    Set the model's total epochs
  */
  void setTotalEpochs(int totalEpochs);
#pragma endregion Total epochs

#pragma region Train metrics
  /*
    Get a copy of the train metrics.
  */
  std::unordered_map<std::string, metricHistoryValue> getTrainMetrics() const;
  /*
    Set the train metrics.
  */
  void
  setTrainMetrics(std::unordered_map<std::string, metricHistoryValue> metrics);
  /*
    Set the train metrics with the metric types.
  */
  void setTrainMetrics(std::vector<std::string> metrics);
#pragma endregion Train metrics

#pragma region Validation metrics
  /*
    Get a copy of the validation metrics.
  */
  std::unordered_map<std::string, metricHistoryValue>
  getValidationMetrics() const;
  /*
    Set the validation metrics.
  */
  void setValidationMetrics(
      std::unordered_map<std::string, metricHistoryValue> metrics);
  /*
    Set the validation metrics with the metric types.
  */
  void setValidationMetrics(std::vector<std::string> metrics);
#pragma endregion Validation metrics
#pragma endregion Properties

#pragma region Load
// TODO
#pragma endregion Load

#pragma region Save
  /*
    Get all the relevant attributes in a serialisable format.

    Attributes:
            - layers -- list of the serialised layers in sequential order
            - loss -- the loss function for the model
            - total_epochs -- the total number of epochs the model has trained
                for
            - train_metrics -- the history of the training metrics for the
                model
            - validation_metrics -- the history of the validation metrics for
                the model

  */
  json toJson() const;
#pragma endregion Save

#pragma region Forward pass
  /*
    Perform the forward pass.
  */
  Eigen::MatrixXd forward(const Eigen::MatrixXd &input);

  /*
    Perform the forward pass and predict the classes for the input.
  */
  std::vector<std::string> predict(const Eigen::MatrixXd &input);
#pragma endregion Forward pass

#pragma region Train
  /*
    Perform the forward pass and store the predictions in the given confusion
    matrix.
  */
  float getLossWithConfusionMatrix(const Eigen::MatrixXd &input,
                                   Eigen::MatrixXi &confusionMatrix,
                                   const std::vector<int> &labels);

private:
  /*
    Perform the training step for one minibatch.
  */
  float trainStep(const Eigen::MatrixXd &data, const std::vector<int> &labels,
                  double learningRate, Eigen::MatrixXi &confusionMatrix);

public:
  /*
    Train the model for the given number of epochs.
  */
  void train(const loader::ImageLoader &loader, double learningRate,
             int batchSize, int epochs);
#pragma endregion Train

#pragma region Test
  /*
    Perform test on the model with the given data loader, returning the loss and
    confusion matrix.
  */
  std::pair<float, Eigen::MatrixXi>
  test(const std::shared_ptr<loader::DatasetBatcher> loader,
       const std::string &indicatorDescription = "");
#pragma endregion Test

#pragma region Metrics
  /*
    Validate the given metric.
  */
  static void validateMetric(const std::string &metric);
  /*
    Validates the given metrics.
  */
  static void validateMetrics(const std::vector<std::string> &metrics);
  /*
    Validates the given metrics.
  */
  static void validateMetrics(
      const std::unordered_map<std::string, metricHistoryValue> &metrics);

  /*
    Convert a vector of metric types to the metric history.
  */
  static std::unordered_map<std::string, metricHistoryValue>
  metricTypesToHistory(const std::vector<std::string> &metrics);

  /*
    Store the metrics.
  */
  static void
  storeMetrics(std::unordered_map<std::string, metricHistoryValue> &metrics,
               Eigen::MatrixXi &confusionMatrix, float loss);

  /*
    Print the tracked metrics.
  */
  static void printMetrics(
      const std::unordered_map<std::string, metricHistoryValue> &metrics,
      const std::vector<std::string> &classes);
#pragma endregion Metrics

#pragma region Visualisation
// TODO
#pragma endregion Visualisation

#pragma region Builtins
  /*
    Perform the forward pass.
  */
  Eigen::MatrixXd operator()(Eigen::MatrixXd input);

  bool operator==(const Model &other) const;
#pragma endregion Builtins
};
} // namespace model