#include "model.hpp"
#include "linear.hpp"
#include "metrics.hpp"
#include "utils/exceptions.hpp"
#include "utils/math.hpp"
#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <indicators/setting.hpp>
#include <unordered_map>
#include <vector>

using namespace model;

#pragma region Keyword args
void Model::KeywordArgs::setTrainMetricsFromMetricTypes(
    std::vector<std::string> metrics) {
  this->trainMetrics = Model::metricTypesToHistory(metrics);
}

void Model::KeywordArgs::setValidationMetricsFromMetricTypes(
    std::vector<std::string> metrics) {
  this->validationMetrics = Model::metricTypesToHistory(metrics);
}
#pragma endregion Keyword args

#pragma region Constructor
Model::Model(std::vector<linear::Linear> layers, loss::CrossEntropyLoss loss,
             KeywordArgs kwargs)
    : layers(layers), loss(loss), totalEpochs(kwargs.totalEpochs),
      classes(kwargs.classes) {
  this->setTotalEpochs(kwargs.totalEpochs);
  this->setTrainMetrics(kwargs.trainMetrics);
  this->setValidationMetrics(kwargs.validationMetrics);
}

Model::Model(std::vector<linear::Linear> layers, loss::CrossEntropyLoss loss)
    : Model(layers, loss, KeywordArgs()) {}
#pragma endregion Constructor

#pragma region Properties
#pragma region Evaluation mode
bool Model::getEval() const { return this->eval; }

void Model::setEval(bool eval) {
  if (this->eval == eval) {
    return;
  }

  for (linear::Linear &layer : this->layers) {
    layer.setEval(eval);
  }
  this->eval = eval;
}
#pragma endregion Evaluation mode

#pragma region Layers

std::vector<linear::Linear> Model::getLayers() const { return this->layers; }

void Model::setLayers(const std::vector<linear::Linear> &layers) {
  if (layers.empty()) {
    throw exceptions::model::EmptyLayersVectorException();
  }
  this->layers = layers;
}
#pragma endregion Layers

#pragma region Loss
loss::CrossEntropyLoss Model::getLoss() const { return this->loss; }

void Model::setLoss(const loss::CrossEntropyLoss &loss) { this->loss = loss; }
#pragma endregion Loss

#pragma region Total epochs
int Model::getTotalEpochs() const { return this->totalEpochs; }

void Model::setTotalEpochs(int totalEpochs) {
  if (totalEpochs < 0) {
    throw exceptions::model::InvalidTotalEpochException(totalEpochs);
  }
  this->totalEpochs = totalEpochs;
}
#pragma endregion Total epochs

#pragma region Train metrics
std::unordered_map<std::string, metricHistoryValue>
Model::getTrainMetrics() const {
  return this->trainMetrics;
}

void Model::setTrainMetrics(
    std::unordered_map<std::string, metricHistoryValue> metrics) {
  Model::validateMetrics(metrics);
  this->trainMetrics = metrics;
}

void Model::setTrainMetrics(std::vector<std::string> metrics) {
  this->setTrainMetrics(Model::metricTypesToHistory(metrics));
}
#pragma endregion Train metrics

#pragma region Validation metrics
std::unordered_map<std::string, metricHistoryValue>
Model::getValidationMetrics() const {
  return this->validationMetrics;
}

void Model::setValidationMetrics(
    std::unordered_map<std::string, metricHistoryValue> metrics) {
  Model::validateMetrics(metrics);
  this->validationMetrics = metrics;
}

void Model::setValidationMetrics(std::vector<std::string> metrics) {
  this->setValidationMetrics(Model::metricTypesToHistory(metrics));
}
#pragma endregion Validation metrics
#pragma endregion Properties

#pragma region Metrics
void Model::validateMetric(const std::string &metric) {
  if (metric != "loss" && !metrics::METRICS.contains(metric)) {
    throw exceptions::model::InvalidMetricException(metric);
  }
}

void Model::validateMetrics(const std::vector<std::string> &metrics) {
  for (const std::string &metric : metrics) {
    Model::validateMetric(metric);
  }
}

void Model::validateMetrics(
    const std::unordered_map<std::string, metricHistoryValue> &metrics) {
  for (const auto &[metric, _] : metrics) {
    Model::validateMetric(metric);
  }
}

std::unordered_map<std::string, metricHistoryValue>
Model::metricTypesToHistory(const std::vector<std::string> &metrics) {
  std::unordered_map<std::string, metricHistoryValue> result;
  Model::validateMetrics(metrics);
  for (const std::string &metric : metrics) {
    if (metrics::SINGLE_VALUE_METRICS.contains(metric)) {
      result[metric] = std::vector<float>();
    } else {
      result[metric] = std::vector<std::vector<float>>();
    }
  }
  return result;
}
#pragma endregion Metrics

#pragma region Forward pass
Eigen::MatrixXd Model::forward(const Eigen::MatrixXd &input) {
  Eigen::MatrixXd out = input;
  for (linear::Linear &layer : this->layers) {
    out = layer(out);
  }
  return out;
}

std::vector<std::string> Model::predict(const Eigen::MatrixXd &input) {
  if (this->classes.empty()) {
    throw exceptions::model::MissingClassesException();
  }
  std::vector<int> predictions =
      utils::math::logitsToPrediction(this->forward(input));
  std::vector<std::string> result(predictions.size());
  for (int i = 0; i < predictions.size(); ++i) {
    result[i] = this->classes[predictions[i]];
  }
  return result;
}
#pragma endregion Forward pass

#pragma region Train
float Model::getLossWithConfusionMatrix(const Eigen::MatrixXd &input,
                                        Eigen::MatrixXi &confusionMatrix,
                                        const std::vector<int> &labels) {
  Eigen::MatrixXd logits = this->forward(input);
  metrics::addToConfusionMatrix(
      confusionMatrix, utils::math::logitsToPrediction(logits), labels);
  return this->loss(logits, labels);
}
#pragma endregion Train

#pragma region Test
std::pair<float, Eigen::MatrixXi>
Model::test(loader::DatasetBatcher loader, std::string indicatorDescription) {
  if (this->classes.empty()) {
    throw exceptions::model::MissingClassesException();
  }

  bool evalMode = this->eval;
  this->setEval(true);

  // Set up indicator
  indicators::show_console_cursor(false);
  indicators::ProgressBar bar{
      indicators::option::BarWidth{50},
      indicators::option::Start{"["},
      indicators::option::Fill{"█"},
      indicators::option::Lead{"█"},
      indicators::option::Remainder{"-"},
      indicators::option::End{"]"},
      indicators::option::PrefixText{indicatorDescription},
      indicators::option::ForegroundColor{indicators::Color::white},
      indicators::option::ShowElapsedTime{true},
      indicators::option::ShowRemainingTime{true},
      indicators::option::ShowPercentage{true},
      indicators::option::MaxProgress{loader.size()}};

  // Perform forward and backward pass
  Eigen::MatrixXi confusionMatrix =
      metrics::getNewConfusionMatrix(this->classes.size());
  float loss = 0;
  for (const auto &[data, labels] : loader) {
    loss += this->getLossWithConfusionMatrix(data, confusionMatrix, labels);
    bar.tick();
  }
  loss /= loader.size();

  // Tear down indicator
  indicators::show_console_cursor(true);

  this->setEval(evalMode);
  return std::make_pair(loss, confusionMatrix);
}
#pragma endregion Test