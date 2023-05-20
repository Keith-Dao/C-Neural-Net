#include "activation_functions.hpp"
#include "linear.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <memory>

using namespace linear;

#pragma region Fixture
struct FixtureData {
  Linear linear;
  Eigen::MatrixXd weight, bias, X, Y, weightGrad, biasGrad;

  FixtureData(const Linear &linear, const Eigen::MatrixXd &weight,
              const Eigen::MatrixXd &bias, const Eigen::MatrixXd &X,
              const Eigen::MatrixXd &Y, const Eigen::MatrixXd &weightGrad,
              const Eigen::MatrixXd &biasGrad)
      : linear(linear), weight(weight), bias(bias), X(X), Y(Y),
        weightGrad(weightGrad), biasGrad(biasGrad){};
};
#pragma endregion Fixture

#pragma region Tests

#pragma endregion Tests

#pragma region Data
Linear layer(3, 2);
Eigen::MatrixXd weight =
    Eigen::VectorXd::LinSpaced(6, 1, 6).reshaped(3, 2).transpose();

#pragma endregion Data