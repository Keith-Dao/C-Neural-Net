#pragma once
#include "cross_entropy_loss.hpp"
#include "linear.hpp"
#include <variant>
#include <vector>

namespace model {
typedef std::variant<std::vector<float>, std::vector<std::vector<float>>>
    metricHistoryValue;

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
    Gets the model's evaluation mode.
  */
  bool getEval() const;
  /*
    Set the model's evaluation mode.
  */
  void setEval(bool eval);
#pragma endregion Evaluation mode

#pragma region Layers
  /*
    Gets the model's layers
  */
  std::vector<linear::Linear> getLayers() const;

  /*
    Sets the model's layers
  */
  void setLayers(const std::vector<linear::Linear> &layers);
#pragma endregion Layers

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
#pragma endregion Metrics
};
} // namespace model