#include "linear.hpp"

using namespace linear;

#pragma region Properties
#pragma region Evaluation mode
bool Linear::getEval() const { return this->eval; }

void Linear::setEval(bool eval) {
  if (this->eval != eval)
    this->input = nullptr;
  this->eval = eval;
}
#pragma endregion Evaluation mode

#pragma region Weight
Eigen::MatrixXd Linear::getWeight() const { return this->weight; }

void Linear::setWeight(Eigen::MatrixXd weight) {
  if (this->weight.rows() != weight.rows() ||
      this->weight.cols() != weight.cols()) {
    throw src_exceptions::InvalidShapeException();
  }
  this->weight = weight;
}
#pragma endregion Weight

#pragma region Bias
Eigen::MatrixXd Linear::getBias() const { return this->bias; };

void Linear::setBias(Eigen::MatrixXd bias) {
  if (this->bias.rows() != bias.rows() || this->bias.cols() != bias.cols()) {
    throw src_exceptions::InvalidShapeException();
  }
  this->bias = bias;
};
#pragma endregion Bias

#pragma region Activation function
std::shared_ptr<activation_functions::ActivationFunction>
Linear::getActivation() const {
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

#pragma region Forward pass
Eigen::MatrixXd Linear::forward(const Eigen::MatrixXd &input) {
  this->input = this->eval ? nullptr : std::make_shared<Eigen::MatrixXd>(input);
  Eigen::MatrixXd output = input * this->weight.transpose();
  output.rowwise() += this->bias.transpose();
  return (*this->activationFunction)(output);
}
#pragma endregion Forward pass
