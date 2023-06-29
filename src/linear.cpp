#include "linear.hpp"
#include "activation_functions.hpp"
#include "exceptions/activation_functions.hpp"
#include "exceptions/differentiable.hpp"
#include "exceptions/eigen.hpp"
#include "exceptions/load.hpp"
#include "utils/matrix.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <map>
#include <math.h>
#include <nlohmann/json.hpp>
#include <typeinfo>
#include <utility>

using namespace linear;

#pragma region Constructor
Linear::Linear(int inChannels, int outChannels, std::string activation)
    : inChannels(inChannels), outChannels(outChannels) {
  if (activation == "NoActivation") {
    this->activationFunction =
        std::make_shared<activation_functions::NoActivation>();
  } else if (activation == "ReLU") {
    this->activationFunction = std::make_shared<activation_functions::ReLU>();
  } else {
    throw exceptions::activation::InvalidActivationException(activation);
  }
  double distributionRange = sqrt(1 / (double)inChannels);
  this->weight =
      Eigen::MatrixXd::Random(outChannels, inChannels) * distributionRange;
  this->bias =
      Eigen::VectorXd::Random(outChannels).transpose() * distributionRange;
}
#pragma endregion Constructor

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
    throw exceptions::eigen::InvalidShapeException(this->weight, weight);
  }
  this->weight = weight;
}
#pragma endregion Weight

#pragma region Bias
Eigen::MatrixXd Linear::getBias() const { return this->bias; };

void Linear::setBias(Eigen::MatrixXd bias) {
  if (this->bias.rows() != bias.rows() || this->bias.cols() != bias.cols()) {
    throw exceptions::eigen::InvalidShapeException(this->bias, bias);
  }
  this->bias = bias;
}
#pragma endregion Bias

#pragma region Activation function
std::shared_ptr<activation_functions::ActivationFunction>
Linear::getActivation() const {
  return this->activationFunction;
}

void Linear::setActivation(std::string activation) {
  if (activation == "ReLU") {
    this->setActivation<activation_functions::ReLU>();
  } else if (activation == "NoActivation") {
    this->setActivation<activation_functions::NoActivation>();
  } else {
    throw exceptions::activation::InvalidActivationException(activation);
  }
}
#pragma endregion Activation function
#pragma endregion Properties

#pragma region Load
Linear Linear::fromJson(const json &values) {
  if (values["class"] != "Linear") {
    throw exceptions::load::InvalidClassAttributeValue();
  }

  Eigen::MatrixXd weight = utils::matrix::fromJson(values["weight"]),
                  bias = utils::matrix::fromJson(json::array({values["bias"]}))
                             .transpose();
  int outChannels = weight.rows(), inChannels = weight.cols();
  Linear layer(inChannels, outChannels);
  layer.setWeight(weight);
  layer.setBias(bias);
  layer.setActivation(values["activation_function"]);
  return layer;
}
#pragma endregion Load

#pragma region Save
json Linear::toJson() const {
  return {{"class", "Linear"},
          {"weight", utils::matrix::toJson(this->weight)},
          {"bias",
           utils::matrix::toJson(
               this->bias.transpose())[0]}, // Needs to match the python output
          {"activation_function", this->activationFunction->getName()}};
}
#pragma endregion Save

#pragma region Forward pass
Eigen::MatrixXd Linear::forward(const Eigen::MatrixXd &input) {
  this->input = this->eval ? nullptr : std::make_shared<Eigen::MatrixXd>(input);
  Eigen::MatrixXd output = input * this->weight.transpose();
  output.rowwise() += this->bias.transpose();
  return (*this->activationFunction)(output);
}
#pragma endregion Forward pass

#pragma region Backward pass
/*
  Perform the backward pass for the layer.
*/
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
Linear::backward(const Eigen::MatrixXd &grad) {
  if (this->eval) {
    throw exceptions::differentiable::BackwardCalledInEvalModeException();
  }
  if (this->input == nullptr) {
    throw exceptions::differentiable::BackwardCalledWithNoInputException();
  }

  Eigen::MatrixXd totalGrad = grad.array() *
                              this->activationFunction->backward().array(),
                  weightGrad = totalGrad.transpose() * *this->input,
                  biasGrad = totalGrad.colwise().sum().transpose(),
                  inputGrad = totalGrad * this->weight;
  return std::make_tuple(inputGrad, weightGrad, biasGrad);
}

Eigen::MatrixXd Linear::update(const Eigen::MatrixXd &grad,
                               const double learningRate) {
  auto [inputGrad, weightGrad, biasGrad] = this->backward(grad);
  this->weight -= learningRate * weightGrad;
  this->bias -= learningRate * biasGrad;
  return inputGrad;
}
#pragma endregion Backward pass

#pragma region Builtins
Eigen::MatrixXd Linear::operator()(const Eigen::MatrixXd &input) {
  return this->forward(input);
}

bool Linear::operator==(const Linear &other) const {
  return typeid(*this) == typeid(other) &&
         this->inChannels == other.inChannels &&
         this->outChannels == other.outChannels &&
         this->weight.isApprox(other.weight) &&
         this->bias.isApprox(other.bias) &&
         *this->activationFunction == *other.activationFunction;
}
#pragma endregion Builtins