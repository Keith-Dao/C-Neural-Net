#pragma once
#include "activation_functions.hpp"
#include <Eigen/Dense>
#include <exception>
#include <math.h>
#include <memory>

namespace linear {
class Linear {
  std::shared_ptr<Eigen::MatrixXd> input = nullptr;
  Eigen::MatrixXd weight, bias;
  std::shared_ptr<activation_functions::ActivationFunction> activationFunction;
  bool eval = false;

  /*
    Create and set the activation function.
  */
  template <typename T> void setActivation() {
    this->activationFunction = std::make_shared<T>();
  };

public:
  int inChannels, outChannels;
  Linear(int inChannels, int outChannels,
         std::shared_ptr<activation_functions::ActivationFunction>
             activationFunction =
                 std::make_shared<activation_functions::NoActivation>())
      : inChannels(inChannels), outChannels(outChannels),
        activationFunction(activationFunction) {
    double distributionRange = sqrt(1 / (double)inChannels);
    this->weight =
        Eigen::MatrixXd::Random(outChannels, inChannels) * distributionRange;
    this->bias =
        Eigen::VectorXd::Random(outChannels).transpose() * distributionRange;
  }

#pragma region Properties
#pragma region Evaluation mode
  /*
    Get the layer's evaluation mode.
  */
  bool getEval() const;
  /*
    Set the layer's evaluation mode.
  */
  void setEval(bool eval);
#pragma endregion Evaluation mode

#pragma region Weight
  /*
    Get the layer's weight.
  */
  Eigen::MatrixXd getWeight() const;
  /*
    Set the layer's weight.
  */
  void setWeight(Eigen::MatrixXd weight);
#pragma endregion Weight

#pragma region Bias
  /*
    Get the layer's bias.
  */
  Eigen::MatrixXd getBias() const;
  /*
    Set the layer's bias.
  */
  void setBias(Eigen::MatrixXd bias);
#pragma endregion Bias

#pragma region Activation function
  /*
    Get the layer's activation function.
  */
  std::shared_ptr<activation_functions::ActivationFunction>
  getActivation() const;
  /*
    Set the layer's activation function.
  */
  void setActivation(std::string activation_function);
#pragma endregion Activation function
#pragma endregion Properties

#pragma region Load
// TODO
#pragma endregion Load

#pragma region Save
// TODO
#pragma endregion Save

#pragma region Forward pass
  /*
    Perform the forward pass for the layer.
  */
  Eigen::MatrixXd forward(const Eigen::MatrixXd &input);
#pragma endregion Forward pass

#pragma region Builtins
  /*
    Perform the forward pass for the layer.
  */
  Eigen::MatrixXd operator()(const Eigen::MatrixXd &input) {
    return this->forward(input);
  }
#pragma endregion Builtins
};

#pragma region Exceptions
class InvalidShapeException : public std::exception {
  virtual const char *what() const throw() {
    return "An invalid shape was provided.";
  }
};
#pragma endregion Exceptions
} // namespace linear
