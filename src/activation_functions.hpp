#pragma once
#include <Eigen/Dense>
#include <any>
#include <exception>

namespace activation_functions {
#pragma region ActivationFunction Abstract class
class ActivationFunction {
protected:
  Eigen::MatrixXd *input = nullptr;

public:
  /*
    Performs the forward pass.
  */
  virtual Eigen::MatrixXd forward(Eigen::MatrixXd &input) = 0;
  /*
    Performs the backward pass.
  */
  virtual Eigen::MatrixXd backward() = 0;

  /*
    Performs the forward pass.
  */
  inline Eigen::MatrixXd operator()(Eigen::MatrixXd &input) {
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
    Performs the forward pass.
  */
  Eigen::MatrixXd forward(Eigen::MatrixXd &input) override;
  /**
    Performs the backward pass.
  */
  Eigen::MatrixXd backward() override;

  // template <typename T> inline bool operator==(const T &other) { return
  // false; } inline bool operator==(const NoActivation &other) { return true; }
  // template <typename T> inline bool operator!=(const T &other) {
  //   return !(this == other);
  // }
};
#pragma endregion NoActivation

#pragma region ReLU
class ReLU : public ActivationFunction {
public:
  /*
    Performs the forward pass.
  */
  Eigen::MatrixXd forward(Eigen::MatrixXd &input) override;
  /*
    Performs the backward pass.
  */
  Eigen::MatrixXd backward() override;

  // template <typename T> inline bool operator==(T &other) { return false; }
  // inline bool operator==(const ReLU &other) { return true; }
  // template <typename T> inline bool operator!=(T &other) {
  //   return !(this == other);
  // }
};
#pragma endregion ReLU

#pragma region Exceptions
class BackwardBeforeForwardException : public std::exception {
  virtual const char *what() const throw() {
    return "backward was called before forward.";
  }
};
#pragma endregion Exceptions
} // namespace activation_functions