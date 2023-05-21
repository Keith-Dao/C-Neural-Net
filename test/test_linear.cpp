#include "activation_functions.hpp"
#include "exceptions.hpp"
#include "linear.hpp"
#include <Eigen/Dense>
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
  layer.setBias(Eigen::VectorXd::LinSpaced(2, 1, 2));
  if (activation != "") {
    layer.setActivation(activation);
  }

  return layer;
};

enum DataSize { small, large, largeWithNegative };
std::string getDataSizeName(DataSize dataSize) {
  switch (dataSize) {
  case small:
    return "small";
  case large:
    return "large";
  case largeWithNegative:
    return "large with negative";
  default:
    throw "Invalid";
  }
}

#pragma region Forward data
typedef std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> forwardData;
forwardData getSmall(const std::string &activation) {
  Eigen::MatrixXd input = Eigen::VectorXd::LinSpaced(3, 1, 3).transpose(),
                  output{{15, 34}};
  return std::make_tuple(input, output);
}
forwardData getLarge(const std::string &activation) {
  Eigen::MatrixXd
      input = Eigen::VectorXd::LinSpaced(30, 1, 30).reshaped(3, 10).transpose(),
      output{{15., 34.},   {33., 79.},   {51., 124.},  {69., 169.},
             {87., 214.},  {105., 259.}, {123., 304.}, {141., 349.},
             {159., 394.}, {177., 439.}};
  return std::make_tuple(input, output);
}
forwardData getLargeWithNegative(const std::string &activation) {
  Eigen::MatrixXd
      input =
          Eigen::VectorXd::LinSpaced(30, -10, 19).reshaped(3, 10).transpose(),
      output = activation == "ReLU"
                   ? Eigen::MatrixXd{{-51., -131.}, {-33., -86.}, {-15., -41.},
                                     {3., 4.},      {21., 49.},   {39., 94.},
                                     {57., 139.},   {75., 184.},  {93., 229.},
                                     {111., 274.}}
                   : Eigen::MatrixXd{{0., 0.},    {0., 0.},    {0., 0.},
                                     {3., 4.},    {21., 49.},  {39., 94.},
                                     {57., 139.}, {75., 184.}, {93., 229.},
                                     {111., 274.}};
  return std::make_tuple(input, output);
}

forwardData getForwardData(DataSize size, std::string activation) {
  switch (size) {
  case small:
    return getSmall(activation);
  case large:
    return getLarge(activation);
  case largeWithNegative:
    return getLargeWithNegative(activation);
  default:
    throw "Invalid";
  }
}
#pragma endregion Forward data

struct FixtureData {
  Linear layer;
  DataSize dataSize;

  FixtureData(std::string activation, DataSize(dataSize))
      : layer(getLayer(activation)), dataSize(dataSize){};
};
std::ostream &operator<<(std::ostream &os, FixtureData const &fixture) {
  return os << fixture.layer.getActivation()->getName() + " - " +
                   getDataSizeName(fixture.dataSize);
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

TEST(Linear, Test_Weight_Invalid_Shape) {
  Linear layer = getLayer();
  Eigen::MatrixXd weight{{1}};
  EXPECT_THROW(layer.setWeight(weight), src_exceptions::InvalidShapeException);
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

TEST(Linear, Test_Bias_Invalid_Shape) {
  Linear layer = getLayer();
  Eigen::MatrixXd bias{{1}};
  EXPECT_THROW(layer.setBias(bias), src_exceptions::InvalidShapeException);
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
  auto [X, Y] =
      getForwardData(GetParam().dataSize, layer.getActivation()->getName());
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
  auto [X, Y] =
      getForwardData(GetParam().dataSize, layer.getActivation()->getName());
  ASSERT_TRUE(Y.isApprox(layer(X))) << "Call:\n"
                                    << layer.getActivation()->getName() << "\n"
                                    << X << "\n"
                                    << Y << "\n";
}
#pragma endregion Builtins
#pragma endregion Tests

#pragma region Data
INSTANTIATE_TEST_SUITE_P(, TestLinear,
                         ::testing::Values(FixtureData("NoActivation", small),
                                           FixtureData("ReLU", small),
                                           FixtureData("NoActivation", large),
                                           FixtureData("ReLU", large)));
#pragma endregion Data
} // namespace test_linear