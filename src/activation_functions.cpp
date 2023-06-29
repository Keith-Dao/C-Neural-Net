#include "activation_functions.hpp"
#include "exceptions/differentiable.hpp"
#include <Eigen/Dense>
#include <algorithm>

using namespace activation_functions;

#pragma region ActivationFunction
Eigen::MatrixXd ActivationFunction::forward(const Eigen::MatrixXd &input) {
  this->input = std::make_shared<Eigen::MatrixXd>(input);
  return input;
}

Eigen::MatrixXd ActivationFunction::backward() {
  if (this->input == nullptr) {
    throw exceptions::differentiable::BackwardBeforeForwardException();
  }
  return *this->input;
}
#pragma endregion ActivationFunction

#pragma region NoActivation
std::string NoActivation::getName() const { return "NoActivation"; };

Eigen::MatrixXd NoActivation::forward(const Eigen::MatrixXd &input) {
  return ActivationFunction::forward(input);
}

Eigen::MatrixXd NoActivation::backward() {
  ActivationFunction::backward();
  return Eigen::MatrixXd::Ones(this->input->rows(), this->input->cols());
}
#pragma endregion NoActivation

#pragma region ReLU
std::string ReLU::getName() const { return "ReLU"; };

Eigen::MatrixXd ReLU::forward(const Eigen::MatrixXd &input) {
  ActivationFunction::forward(input);
  return input.cwiseProduct((input.array() > 0).cast<double>().matrix());
}

Eigen::MatrixXd ReLU::backward() {
  ActivationFunction::backward();
  return (this->input->array() > 0).cast<double>().matrix();
}
#pragma endregion ReLU