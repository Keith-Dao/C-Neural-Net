#include "activation_functions.hpp"
#include "utils/exceptions.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

using namespace activation_functions;

namespace test_activation_functions {
#pragma region Fixture
std::shared_ptr<ActivationFunction>
getActivationFunction(std::string activation) {
  if (activation == "ReLU") {
    return std::make_shared<ReLU>();
  }
  if (activation == "NoActivation") {
    return std::make_shared<NoActivation>();
  }
  throw exceptions::activation::InvalidActivationException(activation);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
getData(std::string activation) {
  if (activation != "ReLU" && activation != "NoActivation") {
    throw exceptions::activation::InvalidActivationException(activation);
  }

  Eigen::MatrixXd
      X = Eigen::VectorXd::LinSpaced(20, -10, 9).reshaped(4, 5).transpose(),
      Y = activation == "ReLU" ? Eigen::MatrixXd{{0, 0, 0, 0},
                                                 {0, 0, 0, 0},
                                                 {0, 0, 0, 1},
                                                 {2, 3, 4, 5},
                                                 {6, 7, 8, 9}}
                               : Eigen::VectorXd::LinSpaced(20, -10, 9)
                                     .reshaped(4, 5)
                                     .transpose(),
      grad = activation == "ReLU" ? Eigen::MatrixXd{{0, 0, 0, 0},
                                                    {0, 0, 0, 0},
                                                    {0, 0, 0, 1},
                                                    {1, 1, 1, 1},
                                                    {1, 1, 1, 1}}
                                  : Eigen::MatrixXd::Ones(5, 4);

  return std::make_tuple(X, Y, grad);
}

struct FixtureData {
  std::string type;
  FixtureData(const std::string &type) : type(type){};
};
// Display a human readable name for the fixture data.
std::ostream &operator<<(std::ostream &os, FixtureData const &fixture) {
  return os << fixture.type;
}

class TestActivationFunctions : public testing::TestWithParam<FixtureData> {};
#pragma endregion Fixture

#pragma region Tests
TEST_P(TestActivationFunctions, TestCall) {
  std::shared_ptr<ActivationFunction> function =
      getActivationFunction(GetParam().type);
  auto [X, Y, _] = getData(GetParam().type);
  decltype(X) originalX(X);
  ASSERT_TRUE((*function)(X).isApprox(Y)) << "Operator:\n"
                                          << typeid(function).name() << "\n"
                                          << X << "\n"
                                          << Y << "\n";
  ASSERT_TRUE(originalX.isApprox(X))
      << "Operation was done inplace but should not have been.";
}

TEST_P(TestActivationFunctions, TestForward) {
  std::shared_ptr<ActivationFunction> function =
      getActivationFunction(GetParam().type);
  auto [X, Y, _] = getData(GetParam().type);
  decltype(X) originalX(X);
  ASSERT_TRUE(function->forward(X).isApprox(Y))
      << "Operator:\n"
      << typeid(function).name() << "\n"
      << X << "\n"
      << Y << "\n";
  ASSERT_TRUE(originalX.isApprox(X))
      << "Operation was done inplace but should not have been.";
}

TEST_P(TestActivationFunctions, TestBackward) {
  std::shared_ptr<ActivationFunction> function =
      getActivationFunction(GetParam().type);
  auto [X, _, grad] = getData(GetParam().type);
  (*function)(X);
  ASSERT_TRUE(function->backward().isApprox(grad))
      << "Backward:\n"
      << typeid(function).name() << "\n"
      << X << "\n"
      << grad << "\n";
}

TEST_P(TestActivationFunctions, TestBackwardBeforeForward) {
  std::shared_ptr<ActivationFunction> function =
      getActivationFunction(GetParam().type);
  EXPECT_THROW(function->backward(),
               exceptions::differentiable::BackwardBeforeForwardException);
}

TEST_P(TestActivationFunctions, TestEqual) {
  std::shared_ptr<ActivationFunction> function =
      getActivationFunction(GetParam().type);
  auto [X, Y, _] = getData(GetParam().type);
  ASSERT_EQ(function->getName() == ReLU().getName(), *function == ReLU());
  ASSERT_EQ(function->getName() == NoActivation().getName(),
            *function == NoActivation());
}
#pragma endregion Tests

#pragma region Data
INSTANTIATE_TEST_SUITE_P(, TestActivationFunctions,
                         ::testing::Values(FixtureData("NoActivation"),
                                           FixtureData("ReLU")));
#pragma endregion Data
} // namespace test_activation_functions