#include "model.hpp"
#include "image_loader.hpp"
#include "linear.hpp"
#include "metrics.hpp"
#include "utils/exceptions.hpp"
#include "utils/indicator.hpp"
#include "utils/math.hpp"
#include "utils/string.hpp"
#include <fstream>
#include <indicators/cursor_control.hpp>
#include <indicators/font_style.hpp>
#include <indicators/progress_bar.hpp>
#include <indicators/setting.hpp>
#include <tabulate/row.hpp>
#include <tabulate/table.hpp>

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
#pragma region Classes
std::vector<std::string> Model::getClasses() const { return this->classes; };

void Model::setClasses(std::vector<std::string> classes) {
  this->classes = classes;
}
#pragma endregion Classes

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

#pragma region Save
json Model::toJson() const {
  auto metricsHistoryToJson =
      [](const std::unordered_map<std::string, metricHistoryValue> &metrics) {
        json result;
        for (const auto &[metric, history] : metrics) {
          json data = json::array();
          if (metrics::SINGLE_VALUE_METRICS.contains(metric)) {
            for (const auto &x : history) {
              data.push_back(std::get<float>(x));
            }
          } else {
            for (const auto &x : history) {
              data.push_back(std::get<std::vector<float>>(x));
            }
          }
          result[metric] = data;
        }
        return result;
      };

  json layers = json::array();
  for (const linear::Linear &layer : this->layers) {
    layers.push_back(layer.toJson());
  }

  return {{"class", "Model"},
          {"layers", layers},
          {"loss", this->loss.toJson()},
          {"total_epochs", this->totalEpochs},
          {"train_metrics", metricsHistoryToJson(this->trainMetrics)},
          {"validation_metrics", metricsHistoryToJson(this->validationMetrics)},
          {"classes", this->classes}};
}

void Model::save(std::string path) const {
  std::filesystem::path savePath(path);
  if (savePath.extension() != ".json") {
    throw exceptions::model::InvalidExtensionException(savePath.extension());
  }

  std::ofstream file(savePath);
  file << this->toJson().dump();
  file.close();
}
#pragma endregion Save

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
  std::cout << " here" << logits.rows() << ", " << labels.size() << std::endl;
  metrics::addToConfusionMatrix(
      confusionMatrix, utils::math::logitsToPrediction(logits), labels);
  return this->loss(logits, labels);
}

float Model::trainStep(const Eigen::MatrixXd &data,
                       const std::vector<int> &labels, double learningRate,
                       Eigen::MatrixXi &confusionMatrix) {
  std::cout << data.rows() << ", " << labels.size() << std::endl;
  float loss = this->getLossWithConfusionMatrix(data, confusionMatrix, labels);
  Eigen::MatrixXd grad = this->loss.backward();
  for (auto it = this->layers.rbegin(); it != this->layers.rend(); ++it) {
    grad = it->update(grad, learningRate);
  }
  return loss;
}

void Model::train(const loader::ImageLoader &loader, double learningRate,
                  int batchSize, int epochs) {
  this->classes = loader.getClasses();
  for (int epoch = 1; epoch < epochs + 1; ++epoch) {
    // Training
    {
      std::shared_ptr<loader::DatasetBatcher> trainingData =
          loader("train", batchSize);
      Eigen::MatrixXi confusionMatrix =
          metrics::getNewConfusionMatrix(this->classes.size());
      float loss = 0;

      indicators::show_console_cursor(false);
      ProgressBar bar = utils::indicators::getDefaultProgressBar();
      bar.set_option(indicators::option::PrefixText{
          "Training epoch " + std::to_string(epoch) + "/" +
          std::to_string(epochs) + ": "});
      bar.set_option(indicators::option::MaxProgress{trainingData->size()});

      for (const auto &[data, labels] : *trainingData) {
        loss += this->trainStep(data, labels, learningRate, confusionMatrix);
        bar.tick();
      }
      loss /= trainingData->size();
      Model::storeMetrics(this->trainMetrics, confusionMatrix, loss);
      Model::printMetrics(this->trainMetrics, this->classes);
      indicators::show_console_cursor(true);
    }

    // Validation
    {
      std::shared_ptr<loader::DatasetBatcher> validationData =
          loader("test", batchSize);
      if (validationData->size() == 0) {
        continue;
      }

      auto [loss, confusionMatrix] = this->test(
          validationData, "Validation epoch " + std::to_string(epoch) + "/" +
                              std::to_string(epochs) + ": ");
      Model::storeMetrics(this->validationMetrics, confusionMatrix, loss);
      Model::printMetrics(this->validationMetrics, this->classes);
    }
  }
  this->totalEpochs += epochs;
}
#pragma endregion Train

#pragma region Test
std::pair<float, Eigen::MatrixXi>
Model::test(const std::shared_ptr<loader::DatasetBatcher> batcher,
            const std::string &indicatorDescription) {
  if (this->classes.empty()) {
    throw exceptions::model::MissingClassesException();
  }

  bool evalMode = this->eval;
  this->setEval(true);

  // Set up indicator
  indicators::show_console_cursor(false);
  indicators::ProgressBar bar = utils::indicators::getDefaultProgressBar();
  bar.set_option(indicators::option::PrefixText{indicatorDescription});
  bar.set_option(indicators::option::MaxProgress{batcher->size()});

  // Perform forward and backward pass
  Eigen::MatrixXi confusionMatrix =
      metrics::getNewConfusionMatrix(this->classes.size());
  float loss = 0;
  for (const auto &[data, labels] : *batcher) {
    loss += this->getLossWithConfusionMatrix(data, confusionMatrix, labels);
    bar.tick();
  }
  loss /= batcher->size();

  // Tear down indicator
  indicators::show_console_cursor(true);

  this->setEval(evalMode);
  return std::make_pair(loss, confusionMatrix);
}
#pragma endregion Test

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
    result[metric] = {};
  }
  return result;
}

void Model::storeMetrics(
    std::unordered_map<std::string, metricHistoryValue> &metrics,
    Eigen::MatrixXi &confusionMatrix, float loss) {
  for (auto &[metric, history] : metrics) {
    if (metric == "loss") {
      history.push_back(loss);
    } else {
      history.push_back(metrics::METRICS.at(metric)(confusionMatrix));
    }
  }
}

void Model::printMetrics(
    const std::unordered_map<std::string, metricHistoryValue> &metrics,
    const std::vector<std::string> &classes) {
  tabulate::Table::Row_t multiclassHeaders{"Class"}, singularHeaders,
      singularData;
  std::vector<std::vector<std::string>> multiclassData{classes};
  int precision = 4;

  // Get the data
  for (const auto &[metric, history] : metrics) {
    std::string header =
        utils::string::join(utils::string::split(metric, "_"), " ");
    header[0] = std::toupper(header[0]);

    if (metrics::SINGLE_VALUE_METRICS.contains(metric)) {
      singularHeaders.push_back(header);
      singularData.push_back(utils::string::floatToString(
          std::get<float>(history.back()), precision));
    } else {
      multiclassHeaders.push_back(header);
      std::vector<std::string> values;
      for (const float &value : std::get<std::vector<float>>(history.back())) {
        values.push_back(utils::string::floatToString(value, precision));
      }
      multiclassData.push_back(values);
    }
  }

  if (!singularHeaders.empty()) {
    tabulate::Table table;

    // Add rows
    table.add_row(singularHeaders);
    table.add_row(singularData);

    // Style table
    table.format()
        .border(" ")
        .corner(" ")
        .font_align(tabulate::FontAlign::right)
        .hide_border_top()
        .hide_border_bottom();
    table.row(0)
        .format()
        .font_style({tabulate::FontStyle::bold})
        .show_border_top();
    table.row(1).format().border_top("-").show_border_top();

    std::cout << table << std::endl;
  }

  if (multiclassHeaders.size() > 1) {
    tabulate::Table table;

    // Add rows
    table.add_row(multiclassHeaders);
    for (int i = 0; i < multiclassData.size(); ++i) {
      if (classes.size() != multiclassData[i].size()) {
        throw exceptions::model::ClassHistoryMismatchException(
            classes.size(), multiclassData[i].size(),
            std::get<std::string>(multiclassHeaders[i]));
      }
    }

    for (int i = 0; i < classes.size(); ++i) {
      tabulate::Table::Row_t row;
      for (const std::vector<std::string> &data : multiclassData) {
        row.push_back(data[i]);
      }
      table.add_row(row);
    }

    // Style table
    table.format()
        .border(" ")
        .corner(" ")
        .font_align(tabulate::FontAlign::right)
        .hide_border_top()
        .hide_border_bottom();
    table.row(0).format().font_style({tabulate::FontStyle::bold});
    table.row(1).format().border_top("-").show_border_top();

    std::cout << table << std::endl;
  }
}
#pragma endregion Metrics

#pragma region Builtins
Eigen::MatrixXd Model::operator()(Eigen::MatrixXd input) {
  return this->forward(input);
}

bool Model::operator==(const Model &other) const {
  return typeid(*this) == typeid(other) && this->layers == other.layers &&
         this->loss == other.loss;
}
#pragma endregion Builtins
