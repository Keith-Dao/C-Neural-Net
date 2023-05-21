#include "activation_functions.hpp"
#include "linear.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <tuple>

using namespace linear;

namespace test_linear {
#pragma region Fixture
Linear getLayer(std::string activation = "") {
  Linear layer(3, 2);
  layer.setWeight(
      Eigen::VectorXd::LinSpaced(6, 1, 6).reshaped(3, 2).transpose());
  layer.setBias(Eigen::VectorXd::LinSpaced(2, 1, 2).transpose());
  if (activation != "") {
    layer.setActivation(activation);
  }

  return layer;
};

typedef std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> forwardData;
forwardData getSmall(const std::string &activation) {
  Eigen::MatrixXd input = Eigen::VectorXd::LinSpaced(3, 1, 3).transpose(),
                  output{{15, 34}};
  return std::make_tuple(input, output);
}

struct FixtureData {
  Linear layer;
  forwardData (*forwardDataGetter)(const std::string &);

  FixtureData(std::string activation,
              forwardData (*forwardDataGetter)(const std::string &))
      : layer(getLayer(activation)), forwardDataGetter(forwardDataGetter){};
};
std::ostream &operator<<(std::ostream &os, FixtureData const &fixture) {
  return os << fixture.layer.getActivation()->getName();
}
class TestLinear : public testing::TestWithParam<FixtureData> {};
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
  Eigen::MatrixXd bias = Eigen::VectorXd::Ones(layer.outChannels).transpose();
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

#pragma region Forward pass
TEST_P(TestLinear, Test_Forward) {
  Linear layer = GetParam().layer;
  auto [X, Y] = GetParam().forwardDataGetter(layer.getActivation()->getName());
  ASSERT_TRUE(Y.isApprox(layer.forward(X)))
      << "Forward:\n"
      << layer.getActivation()->getName() << "\n"
      << X << "\n"
      << Y << "\n";
}
#pragma endregion Forward pass

#pragma region Builtins
TEST_P(TestLinear, Test_Call) {
  Linear layer = GetParam().layer;
  auto [X, Y] = GetParam().forwardDataGetter(layer.getActivation()->getName());
  ASSERT_TRUE(Y.isApprox(layer(X))) << "Call:\n"
                                    << layer.getActivation()->getName() << "\n"
                                    << X << "\n"
                                    << Y << "\n";
}
#pragma endregion Builtins
#pragma endregion Tests

#pragma region Data
FixtureData noActivationSmall("NoActivation", getSmall),
    reluSmall("ReLU", getSmall);
INSTANTIATE_TEST_SUITE_P(, TestLinear,
                         ::testing::Values(noActivationSmall, reluSmall));
#pragma endregion Data
} // namespace test_linear