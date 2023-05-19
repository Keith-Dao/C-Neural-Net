#include "linear.hpp"

using namespace linear;

#pragma region Properties
#pragma region Evaluation mode
bool Linear::getEval() { return this->eval; }

void Linear::setEval(bool eval) {
  if (this->eval != eval)
    this->input = nullptr;
  this->eval = eval;
}
#pragma endregion Evaluation mode

#pragma region Weight
Eigen::MatrixXd Linear::getWeight() { return this->weight; }

void Linear::setWeight(Eigen::MatrixXd weight) {
  if (this->weight.rows() != weight.rows() ||
      this->weight.cols() != weight.cols()) {
    throw "Invalid shape for new weight.";
  }
  this->weight = weight;
}
#pragma endregion Weight

#pragma region Bias
Eigen::MatrixXd Linear::getBias() { return this->bias; };

void Linear::setBias(Eigen::MatrixXd bias) {
  if (this->bias.rows() != bias.rows() || this->bias.cols() != bias.cols()) {
    throw "Invalid shape for new bias.";
  }
  this->bias = bias;
};
#pragma endregion Bias

#pragma region Activation function
std::shared_ptr<activation_functions::ActivationFunction>
Linear::getActivation() {
  return this->activationFunction;
};

void Linear::setActivation(std::string activation_function) {
  if (activation_function == "ReLU") {
    this->setActivation<activation_functions::ReLU>();
  } else if (activation_function == "NoActivation") {
    this->setActivation<activation_functions::NoActivation>();
  } else {
    throw "Unknown activation function.";
  }
}
#pragma endregion Activation function
#pragma endregion Properties