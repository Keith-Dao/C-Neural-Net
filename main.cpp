#include "cross_entropy_loss.hpp"
#include "image_loader.hpp"
#include "linear.hpp"
#include "model.hpp"
#include "utils/cli.hpp"
#include <iostream>
#include <memory>
#include <readline/history.h>
#include <readline/readline.h>
#include <stdexcept>
#include <tabulate/table.hpp>
#include <yaml-cpp/yaml.h>

#pragma region Helper
namespace utils::yaml {
/*
  Checks whether a YAML node has a value.
*/
bool hasValue(const YAML::Node &node) {
  return node.IsDefined() && !node.IsNull();
}
} // namespace utils::yaml
#pragma endregion Helper

#pragma region Args
struct Args {
  std::string configFile = "config.yaml";
  bool prediction = false;
};

/*
  Display the help text and exit.
*/
void displayHelpText() {
  std::cout << "usage: NeuralNetwork [-h] [-p] [config_file]"
            << "\n\n";
  std::cout << "Neural network for classifying images of digits."
            << "\n\n";

  tabulate::Table table;
  table.add_row({"positional arguments:", ""});
  table.add_row({"config_file", "Path to the config file."});
  table.add_row({"", ""});
  table.add_row({"options:", ""});
  table.add_row({"-h, --help", "show this help message and exit"});
  table.add_row({"-p, --prediction-mode", "Skip to the prediction mode."});

  table.format().border("").corner("").padding_left(4);
  std::vector<int> headerRows{0, 3};
  for (int row : headerRows) {
    table.row(row).format().padding_left(0);
  }

  std::cout << table << "\n" << std::endl;
  exit(0);
}

/*
  Parse the arguments ang get the appropriate arguments.
*/
Args parseArgs(int argc, char **argv) {
  Args args;
  bool configFileProvided = false;
  for (int i = 1; i < argc; ++i) {
    std::string token(argv[i]);
    if (token == "--help" || token == "-h") {
      displayHelpText();
    } else if (token == "-p" || token == "--prediction-mode") {
      args.prediction = true;
    } else {
      if (configFileProvided) {
        displayHelpText();
      }
      configFileProvided = true;
      args.configFile = token;
    }
  }
  if (!configFileProvided) {
    utils::cli::printWarning(
        "Config file path was not provided. Defaulting to config.yaml");
  }
  return args;
}
#pragma endregion Args

#pragma region Config
/*
  Get the config values from the config file.
*/
YAML::Node getConfig(const std::string &configPath) {
  return YAML::LoadFile(configPath);
}

/*
  Get the train validation split from the config file.
*/
float getTrainValidationSplit(const YAML::Node &config) {
  if (!utils::yaml::hasValue(config["train_validation_split"])) {
    utils::cli::printWarning(
        "No value for train_validation_split was provided. Defaulting to 0.7");
    return 0.7;
  }
  return config["train_validation_split"].as<float>();
}

/*
  Get the file formats from the config file.
*/
std::vector<std::string> getFileFormats(const YAML::Node &config) {
  if (!utils::yaml::hasValue(config["file_formats"])) {
    utils::cli::printWarning("No value for file_formats was provided. "
                             "Defaulting to only accept .png");
    return {".png"};
  }
  return config["file_formats"].as<std::vector<std::string>>();
}

/*
  Get the batch size from the config file.
*/
int getBatchSize(const YAML::Node &config) {
  if (!utils::yaml::hasValue(config["batch_size"])) {
    utils::cli::printWarning("Value of batch_size not found, defaulting to 1.");
    return 1;
  }
  int batchSize = config["batch_size"].as<int>();
  if (batchSize <= 0) {
    throw std::invalid_argument("batch_size must be greater than 0.");
  }
  return batchSize;
}
#pragma endregion Config

#pragma region Load model
/*
  Loads the model using the file provided in the config, or use the default
  model if no file is provided.
*/
model::Model getModel(const YAML::Node &config) {
  if (utils::yaml::hasValue(config["model_path"])) {
    return model::Model::load(config["model_path"].as<std::string>());
  }

  // Load the default model
  utils::cli::printWarning(
      "No model file was provided. Loading untrained model.");
  std::vector<linear::Linear> layers{linear::Linear(784, 250, "ReLU"),
                                     linear::Linear(250, 250, "ReLU"),
                                     linear::Linear(250, 10)};
  loss::CrossEntropyLoss loss;
  model::Model::KeywordArgs kwargs;
  kwargs.setTrainMetricsFromMetricTypes(
      utils::yaml::hasValue(config["train_metrics"])
          ? config["train_metrics"].as<std::vector<std::string>>()
          : std::vector<std::string>());
  kwargs.setValidationMetricsFromMetricTypes(
      utils::yaml::hasValue(config["validation_metrics"])
          ? config["validation_metrics"].as<std::vector<std::string>>()
          : std::vector<std::string>());

  return model::Model(layers, loss, kwargs);
}
#pragma endregion Load model

#pragma region Image loader
/*
  Create an image loader.
*/
std::shared_ptr<loader::ImageLoader>
getImageLoader(const YAML::Node &config, const std::string &dataset) {
  if (!utils::yaml::hasValue(config[dataset + "_path"])) {
    utils::cli::printWarning("No value for " + dataset +
                             "_path was provided. Skipping " + dataset +
                             "ing.");
    return nullptr;
  }

  float trainValidationSplit =
      dataset == "test" ? 0 : getTrainValidationSplit(config);
  std::vector<std::string> fileFormats = getFileFormats(config);
  return std::make_shared<loader::ImageLoader>(
      config[dataset + "_path"].as<std::string>(),
      loader::ImageLoader::standardPreprocessing, fileFormats,
      trainValidationSplit);
}
#pragma endregion Image loader

#pragma region Clean up
/*
  Free all the initalized memory used for readline's history.
*/
void cleanUpHistory() {
  HISTORY_STATE *history = history_get_history_state();
  HIST_ENTRY **historyList = history_list();
  for (int i = 0; i < history->length; i++) {
    free_history_entry(historyList[i]);
  }
  free(history);
  free(historyList);
}
#pragma region Clean up

int main(int argc, char **argv) {
  Args args = parseArgs(argc, argv);
  YAML::Node config = getConfig(args.configFile);
  model::Model model = getModel(config);
  using_history();

  cleanUpHistory();
  return 0;
}