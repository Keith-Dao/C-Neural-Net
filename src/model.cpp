#include "model.hpp"
#include "metrics.hpp"
#include "utils/exceptions.hpp"
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
  if (this->totalEpochs < 0) {
    throw exceptions::model::InvalidTotalEpochException(this->totalEpochs);
  }
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
