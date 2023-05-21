#include "activation_functions.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <memory>

using namespace activation_functions;

namespace test_activation_functions {
#pragma region Fixture
struct FixtureData {
  std::shared_ptr<ActivationFunction> function;
  Eigen::MatrixXd X, Y, grad;

  FixtureData(const std::shared_ptr<ActivationFunction> &function,
              const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y,
              const Eigen::MatrixXd &grad)
      : function(function), X(X), Y(Y), grad(grad){};
};
// Display a human readable name for the fixture data.
std::ostream &operator<<(std::ostream &os, FixtureData const &fixture) {
  return os << fixture.function->getName();
}

class TestActivationFunctions : public testing::TestWithParam<FixtureData> {};
#pragma endregion Fixture

#pragma region Tests
TEST_P(TestActivationFunctions, Test_Call) {
  auto [function, X, Y, _] = GetParam();
  decltype(X) originalX(X);
  ASSERT_TRUE((*function)(X).isApprox(Y)) << "Operator:\n"
                                          << typeid(function).name() << "\n"
                                          << X << "\n"
                                          << Y << "\n";
  ASSERT_TRUE(originalX.isApprox(X))
      << "Operation was done inplace but should not have been.";
}

TEST_P(TestActivationFunctions, Test_Forward) {
  auto [function, X, Y, _] = GetParam();
  decltype(X) originalX(X);
  ASSERT_TRUE(function->forward(X).isApprox(Y))
      << "Operator:\n"
      << typeid(function).name() << "\n"
      << X << "\n"
      << Y << "\n";
  ASSERT_TRUE(originalX.isApprox(X))
      << "Operation was done inplace but should not have been.";
}

TEST_P(TestActivationFunctions, Test_Backward) {
  auto [function, X, Y, grad] = GetParam();
  (*function)(X);
  ASSERT_TRUE(function->backward().isApprox(grad))
      << "Backward:\n"
      << typeid(function).name() << "\n"
      << X << "\n"
      << grad << "\n";
}

TEST_P(TestActivationFunctions, Test_Backward_Before_Forward) {
  std::shared_ptr<ActivationFunction> function = GetParam().function;
  EXPECT_THROW(function->backward(), BackwardBeforeForwardException);
}

TEST_P(TestActivationFunctions, Test_Equal) {
  std::shared_ptr<ActivationFunction> function = GetParam().function;
  ASSERT_EQ(function->getName() == ReLU().getName(), *function == ReLU());
  ASSERT_EQ(function->getName() == NoActivation().getName(),
            *function == NoActivation());
}
#pragma endregion Tests

#pragma region Data
FixtureData noActivationData(
    std::make_shared<NoActivation>(),
    Eigen::VectorXd::LinSpaced(20, -10, 9).reshaped(4, 5).transpose(),
    Eigen::VectorXd::LinSpaced(20, -10, 9).reshaped(4, 5).transpose(),
    Eigen::MatrixXd::Ones(5, 4)),
    reluData(std::make_shared<ReLU>(),
             Eigen::VectorXd::LinSpaced(20, -10, 9).reshaped(4, 5).transpose(),
             Eigen::MatrixXd{{0, 0, 0, 0},
                             {0, 0, 0, 0},
                             {0, 0, 0, 1},
                             {2, 3, 4, 5},
                             {6, 7, 8, 9}},
             Eigen::MatrixXd{
                 {0, 0, 0, 0},
                 {0, 0, 0, 0},
                 {0, 0, 0, 1},
                 {1, 1, 1, 1},
                 {1, 1, 1, 1},
             });

INSTANTIATE_TEST_SUITE_P(, TestActivationFunctions,
                         ::testing::Values(noActivationData, reluData));
#pragma endregion Data
} // namespace test_activation_functions