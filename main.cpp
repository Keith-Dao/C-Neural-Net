#include "src/utils/cli.hpp"
#include <iostream>
#include <readline/history.h>
#include <readline/readline.h>
#include <tabulate/table.hpp>
#include <yaml-cpp/yaml.h>

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
#pragma endregion Config

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
  using_history();

  cleanUpHistory();
  return 0;
}