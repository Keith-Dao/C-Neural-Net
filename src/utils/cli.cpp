#include "cli.hpp"
#include <cctype>
#include <cstring>
#include <iostream>
#include <readline/history.h>
#include <readline/readline.h>
#include <termcolor/termcolor.hpp>

bool utils::cli::getIsYesResponse(const std::string &initialMessage) {
  rl_unbind_key('\t');
  std::string response;
  auto getResponse = [&](const char *message) {
    char *buffer = readline(message);
    add_history(buffer);
    response = std::string(buffer);
    free(buffer);
    if (!response.empty()) {
      response[0] = std::tolower(response[0]);
    }
  };

  getResponse(initialMessage.c_str());
  while (response != "y" && response != "n") {
    getResponse("Please enter either y for yes or no for no: ");
  }
  rl_bind_key('\t', rl_complete);
  return response == "y";
}

void utils::cli::printWarning(const std::string &message) {
  std::cout << termcolor::yellow << message << termcolor::reset << std::endl;
}
