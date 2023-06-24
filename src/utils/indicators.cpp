#include "indicator.hpp"

ProgressBar utils::indicators::getDefaultProgressBar() {
  return ProgressBar{option::BarWidth{50},
                     option::Start{"["},
                     option::Fill{"█"},
                     option::Lead{"█"},
                     option::Remainder{"-"},
                     option::End{"]"},
                     option::ForegroundColor{Color::white},
                     option::ShowElapsedTime{true},
                     option::ShowRemainingTime{true},
                     option::ShowPercentage{true}};
}