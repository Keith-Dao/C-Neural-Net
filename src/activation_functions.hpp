#pragma once
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <typeinfo>

namespace activation_functions {
#pragma region ActivationFunction Abstract class
class ActivationFunction {
protected:
  std::shared_ptr<Eigen::MatrixXd> input = nullptr;

public:
  virtual ~ActivationFunction(){};

  /*
    Gets the name for the class.
  */
  virtual std::string getName() const { return "Activation function"; };
  /*
    Performs the forward pass.
  */
  virtual Eigen::MatrixXd forward(const Eigen::MatrixXd &input) = 0;
  /*
    Performs the backward pass.
  */
  virtual Eigen::MatrixXd backward() = 0;

  /*
    Performs the forward pass.
  */
  Eigen::MatrixXd operator()(Eigen::MatrixXd &input) {
    return this->forward(input);
  };

  virtual bool operator==(const ActivationFunction &other) const {
    return typeid(*this) == typeid(other);
  }
};

#pragma endregion ActivationFunction Abstract class

#pragma region NoActivation
class NoActivation : public ActivationFunction {
public:
  /*
    Gets the name for the class.
  */
  std::string getName() const override;
  /*
    Performs the forward pass.
  */
  Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;
  /**
    Performs the backward pass.
  */
  Eigen::MatrixXd backward() override;
};
#pragma endregion NoActivation

#pragma region ReLU
class ReLU : public ActivationFunction {
public:
  /*
    Gets the name for the class.
  */
  std::string getName() const override;
  /*
    Performs the forward pass.
  */
  Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;
  /*
    Performs the backward pass.
  */
  Eigen::MatrixXd backward() override;
};
#pragma endregion ReLU
} // namespace activation_functions