#include "cli.hpp"

bool utils::cli::isYes(std::string response) {
  if (!response.empty()) {
    response[0] = std::tolower(response[0]);
  }
  while (response != "y" && response != "n") {
    std::cout << "Please enter either y for yes or no for no: " << std::flush;
    std::cin >> response;
    if (!response.empty()) {
      response[0] = std::tolower(response[0]);
    }
  }
  return response == "y";
}