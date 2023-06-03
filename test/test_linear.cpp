#include "activation_functions.hpp"
#include "exceptions.hpp"
#include "linear.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <tuple>

using namespace linear;
using json = nlohmann::json;

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
      output = activation != activation_functions::ReLU().getName()
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

#pragma region Grad data
typedef std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
                   Eigen::MatrixXd>
    gradData;
/*
  Gradients with respect the to the output, input, weight and bias respectively.
*/
gradData getSmallGrad(const std::string &activation) {
  Eigen::MatrixXd output = Eigen::MatrixXd::Ones(1, 2), input{{5, 7, 9}},
                  weight{{1, 2, 3}, {1, 2, 3}}, bias{{1}, {1}};
  return std::make_tuple(output, input, weight, bias);
}
/*
  Gradients with respect the to the output, input, weight and bias respectively.
*/
gradData getLargeGrad(const std::string &activation) {
  Eigen::MatrixXd output = Eigen::MatrixXd::Ones(10, 2),
                  input{{5., 7., 9.}, {5., 7., 9.}, {5., 7., 9.}, {5., 7., 9.},
                        {5., 7., 9.}, {5., 7., 9.}, {5., 7., 9.}, {5., 7., 9.},
                        {5., 7., 9.}, {5., 7., 9.}},
                  weight{{145., 155., 165.}, {145., 155., 165.}},
                  bias{{10}, {10}};
  return std::make_tuple(output, input, weight, bias);
}
/*
  Gradients with respect the to the output, input, weight and bias respectively.
*/
gradData getLargeWithNegativeGrad(const std::string &activation) {
  Eigen::MatrixXd output = Eigen::MatrixXd::Ones(10, 2),
                  input = activation == activation_functions::ReLU().getName()
                              ? Eigen::MatrixXd{{0., 0., 0.}, {0., 0., 0.},
                                                {0., 0., 0.}, {5., 7., 9.},
                                                {5., 7., 9.}, {5., 7., 9.},
                                                {5., 7., 9.}, {5., 7., 9.},
                                                {5., 7., 9.}, {5., 7., 9.}}
                              : Eigen::MatrixXd{{5., 7., 9.}, {5., 7., 9.},
                                                {5., 7., 9.}, {5., 7., 9.},
                                                {5., 7., 9.}, {5., 7., 9.},
                                                {5., 7., 9.}, {5., 7., 9.},
                                                {5., 7., 9.}, {5., 7., 9.}},
                  weight =
                      activation == activation_functions::ReLU().getName()
                          ? Eigen::MatrixXd{{56., 63., 70.}, {56., 63., 70.}}
                          : Eigen::MatrixXd{{35., 45., 55.}, {35., 45., 55.}},
                  bias = activation == activation_functions::ReLU().getName()
                             ? Eigen::MatrixXd{{7}, {7}}
                             : Eigen::MatrixXd{{10}, {10}};
  return std::make_tuple(output, input, weight, bias);
}

gradData getGradData(DataSize size, std::string activation) {
  switch (size) {
  case small:
    return getSmallGrad(activation);
  case large:
    return getLargeGrad(activation);
  case largeWithNegative:
    return getLargeWithNegativeGrad(activation);
  default:
    throw "Invalid";
  }
}
#pragma endregion Grad data
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
TEST(Linear, TestEval) {
  Linear linear = getLayer();
  ASSERT_FALSE(linear.getEval());
  linear.setEval(true);
  ASSERT_TRUE(linear.getEval());
}
#pragma endregion Evaluation mode

#pragma region Weight
TEST(Linear, TestWeight) {
  Linear layer = getLayer();
  Eigen::MatrixXd weight =
      Eigen::MatrixXd::Ones(layer.outChannels, layer.inChannels);
  ASSERT_FALSE(weight.isApprox(layer.getWeight()));
  layer.setWeight(weight);
  ASSERT_TRUE(weight.isApprox(layer.getWeight()));
}

TEST(Linear, TestWeight_Invalid_Shape) {
  Linear layer = getLayer();
  Eigen::MatrixXd weight{{1}};
  EXPECT_THROW(layer.setWeight(weight), src_exceptions::InvalidShapeException);
}
#pragma endregion Weight

#pragma region Bias
TEST(Linear, TestBias) {
  Linear layer = getLayer();
  Eigen::MatrixXd bias = Eigen::VectorXd::Ones(layer.outChannels);
  ASSERT_FALSE(bias.isApprox(layer.getBias()));
  layer.setBias(bias);
  ASSERT_TRUE(bias.isApprox(layer.getBias()));
}

TEST(Linear, TestBiasInvalidShape) {
  Linear layer = getLayer();
  Eigen::MatrixXd bias{{1}};
  EXPECT_THROW(layer.setBias(bias), src_exceptions::InvalidShapeException);
}
#pragma endregion Bias

#pragma region Activation function
TEST(Linear, TestActivationFunction) {
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

#pragma region Load
TEST(Linear, TestFromJson) {
  Linear expected = getLayer();
  json values{{"class", "Linear"},
              {"weight", {{1, 2, 3}, {4, 5, 6}}},
              {"bias", {1, 2}},
              {"activation_function", "NoActivation"}};
  ASSERT_EQ(expected, Linear::fromJson(values));

  expected = getLayer("ReLU");
  values = json{{"class", "Linear"},
                {"weight", {{1, 2, 3}, {4, 5, 6}}},
                {"bias", {1, 2}},
                {"activation_function", "ReLU"}};
  ASSERT_EQ(expected, Linear::fromJson(values));
}

TEST(Linear, TestFromJsonInvalidClassAttribute) {
  json values{{"class", "NotLinear"},
              {"weight", {{1, 2, 3}, {4, 5, 6}}},
              {"bias", {1, 2}},
              {"activation_function", "NoActivation"}};
  EXPECT_THROW(Linear::fromJson(values),
               src_exceptions::InvalidClassAttributeValue);
}
#pragma endregion Load

#pragma region Save
TEST(Linear, TestToJson) {
  Linear layer = getLayer();
  json expected{{"class", "Linear"},
                {"weight", {{1, 2, 3}, {4, 5, 6}}},
                {"bias", {1, 2}},
                {"activation_function", "NoActivation"}};
  ASSERT_EQ(expected, layer.toJson()) << "NoActivation layer.\n";

  layer = getLayer("ReLU");
  expected = json{{"class", "Linear"},
                  {"weight", {{1, 2, 3}, {4, 5, 6}}},
                  {"bias", {1, 2}},
                  {"activation_function", "ReLU"}};
  ASSERT_EQ(expected, layer.toJson()) << "ReLU layer.\n";
}
#pragma endregion Save

#pragma region Forward pass
TEST_P(TestLinear, TestForward) {
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

#pragma region Backward pass
TEST_P(TestLinear, TestBackward) {
  Linear layer = GetParam().layer;
  auto [X, _] =
      getForwardData(GetParam().dataSize, layer.getActivation()->getName());
  layer(X);
  auto [grad, trueInputGrad, trueWeightGrad, trueBiasGrad] =
      getGradData(GetParam().dataSize, layer.getActivation()->getName());
  auto [inputGrad, weightGrad, biasGrad] = layer.backward(grad);
  ASSERT_TRUE(trueInputGrad.isApprox(inputGrad));
  ASSERT_TRUE(trueWeightGrad.isApprox(weightGrad));
  ASSERT_TRUE(trueBiasGrad.isApprox(biasGrad));
}

TEST_P(TestLinear, TestBackwardWithEval) {
  Linear layer = GetParam().layer;
  auto [X, _] =
      getForwardData(GetParam().dataSize, layer.getActivation()->getName());
  layer.setEval(true);
  layer(X);
  auto [grad, trueInputGrad, trueWeightGrad, trueBiasGrad] =
      getGradData(GetParam().dataSize, layer.getActivation()->getName());
  EXPECT_THROW(layer.backward(grad),
               src_exceptions::BackwardCalledInEvalModeException);
}

TEST_P(TestLinear, TestBackwardWithNoInput) {
  Linear layer = GetParam().layer;
  auto [grad, trueInputGrad, trueWeightGrad, trueBiasGrad] =
      getGradData(GetParam().dataSize, layer.getActivation()->getName());
  EXPECT_THROW(layer.backward(grad),
               src_exceptions::BackwardCalledWithNoInputException);
}

TEST_P(TestLinear, TestUpdate) {
  Linear layer = GetParam().layer;
  auto [X, _] =
      getForwardData(GetParam().dataSize, layer.getActivation()->getName());
  layer(X);
  auto [grad, trueInputGrad, trueWeightGrad, trueBiasGrad] =
      getGradData(GetParam().dataSize, layer.getActivation()->getName());
  double learningRate = 1e-4;
  Eigen::MatrixXd trueWeight = Eigen::VectorXd::LinSpaced(6, 1, 6)
                                   .reshaped(3, 2)
                                   .transpose(),
                  trueBias = Eigen::VectorXd::LinSpaced(2, 1, 2);
  ASSERT_TRUE(trueWeight.isApprox(layer.getWeight()));
  ASSERT_TRUE(trueBias.isApprox(layer.getBias()));

  Eigen::MatrixXd inputGrad = layer.update(grad, learningRate);
  ASSERT_TRUE(trueInputGrad.isApprox(inputGrad));
  trueWeight -= learningRate * trueWeightGrad;
  trueBias -= learningRate * trueBiasGrad;
  ASSERT_TRUE(trueWeight.isApprox(layer.getWeight()));
  ASSERT_TRUE(trueBias.isApprox(layer.getBias()));
}
#pragma endregion Backward pass

#pragma region Builtins
TEST_P(TestLinear, TestCall) {
  Linear layer = GetParam().layer;
  auto [X, Y] =
      getForwardData(GetParam().dataSize, layer.getActivation()->getName());
  ASSERT_TRUE(Y.isApprox(layer(X))) << "Call:\n"
                                    << layer.getActivation()->getName() << "\n"
                                    << X << "\n"
                                    << Y << "\n";
}

TEST(Linear, TestEqual) {
  Linear layer = getLayer(), other = getLayer();
  ASSERT_EQ(layer, other) << "Layers should be the same.\n";

  // Different shape
  other = Linear(2, 3);
  ASSERT_NE(layer, other) << "Layers have different shape.\n";

  // Different weight
  other = getLayer();
  other.setWeight(Eigen::MatrixXd::Ones(2, 3));
  ASSERT_NE(layer, other) << "Layers have different weights.\n";

  // Different bias
  other = getLayer();
  other.setBias(Eigen::VectorXd::Ones(2));
  ASSERT_NE(layer, other) << "Layers have different bias.\n";

  // Different activation
  other = getLayer("ReLU");
  ASSERT_NE(layer, other) << "Layers have different activation functions.\n";
}
#pragma endregion Builtins
#pragma endregion Tests

#pragma region Data
INSTANTIATE_TEST_SUITE_P(
    , TestLinear,
    ::testing::Values(FixtureData("NoActivation", small),
                      FixtureData("ReLU", small),
                      FixtureData("NoActivation", large),
                      FixtureData("ReLU", large),
                      FixtureData("NoActivation", largeWithNegative),
                      FixtureData("ReLU", largeWithNegative)));
#pragma endregion Data
} // namespace test_linear