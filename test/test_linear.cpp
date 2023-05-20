#include "activation_functions.hpp"
#include "linear.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

using namespace linear;

namespace test_linear {
#pragma region Fixture
Linear getLayer(std::string activation = "") {
  Linear layer(3, 2);
  layer.setWeight(
      Eigen::VectorXd::LinSpaced(6, 1, 6).reshaped(3, 2).transpose());
  layer.setBias(Eigen::VectorXd::LinSpaced(2, 1, 2));
  if (activation != "") {
    layer.setActivation(activation);
  }

  return layer;
};

struct FixtureData {
  Linear layer;
  Eigen::MatrixXd X, Y, weightGrad, biasGrad;

  FixtureData(std::string activation, const Eigen::MatrixXd &X,
              const Eigen::MatrixXd &Y, const Eigen::MatrixXd &weightGrad,
              const Eigen::MatrixXd &biasGrad)
      : layer(getLayer(activation)), X(X), Y(Y), weightGrad(weightGrad),
        biasGrad(biasGrad){};
};
std::ostream &operator<<(std::ostream &os, FixtureData const &fixture) {
  return os << fixture.layer.getActivation()->getName();
}
#pragma endregion Fixture

#pragma region Tests
#pragma region Properties
#pragma region Evaluation mode
TEST(Linear, Test_Eval) {
  Linear linear = getLayer();
  ASSERT_FALSE(linear.getEval());
  linear.setEval(true);
  ASSERT_TRUE(linear.getEval());
}
#pragma endregion Evaluation mode

#pragma region Weight
TEST(Linear, Test_Weight) {
  Linear layer = getLayer();
  Eigen::MatrixXd weight =
      Eigen::MatrixXd::Ones(layer.outChannels, layer.inChannels);
  ASSERT_FALSE(weight.isApprox(layer.getWeight()));
  layer.setWeight(weight);
  ASSERT_TRUE(weight.isApprox(layer.getWeight()));
}
#pragma endregion Weight

#pragma region Bias
TEST(Linear, Test_Bias) {
  Linear layer = getLayer();
  Eigen::MatrixXd bias = Eigen::VectorXd::Ones(layer.outChannels);
  ASSERT_FALSE(bias.isApprox(layer.getBias()));
  layer.setBias(bias);
  ASSERT_TRUE(bias.isApprox(layer.getBias()));
}
#pragma endregion Bias

#pragma region Activation function
TEST(Linear, Test_Activation_Function) {
  Linear layer = getLayer();
  ASSERT_EQ(*layer.getActivation(), activation_functions::NoActivation());
  layer.setActivation("ReLU");
  ASSERT_EQ(*layer.getActivation(), activation_functions::ReLU());

  layer = getLayer("ReLU");
  ASSERT_EQ(*layer.getActivation(), activation_functions::ReLU());
  layer.setActivation("NoActivation");
  ASSERT_EQ(*layer.getActivation(), activation_functions::NoActivation());
}
#pragma endregion Activation function
#pragma endregion Properties
#pragma endregion Tests

#pragma region Data

#pragma endregion Data
} // namespace test_linear