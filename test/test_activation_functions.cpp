#include "activation_functions.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>

using namespace activation_functions;

#pragma region NoActivation
class TestNoActivation : public ::testing::Test {
protected:
  NoActivation noActivation;
  Eigen::MatrixXd
      X = Eigen::VectorXd::LinSpaced(20, -10, 9).reshaped(4, 5).transpose(),
      Y = Eigen::MatrixXd(X), grad = Eigen::MatrixXd::Ones(X.rows(), X.cols());
};

TEST_F(TestNoActivation, Test_Call) {
  ASSERT_TRUE(noActivation(X).isApprox(Y)) << "Operator:\n"
                                           << X << "\n"
                                           << Y << "\n";
}

TEST_F(TestNoActivation, Test_Forward) {
  ASSERT_TRUE(noActivation.forward(X).isApprox(Y)) << "Forward:\n"
                                                   << X << "\n"
                                                   << Y << "\n";
}

TEST_F(TestNoActivation, Test_Backward) {
  noActivation(X);
  ASSERT_TRUE(noActivation.backward().isApprox(grad)) << "Backward:\n"
                                                      << X << "\n"
                                                      << Y << "\n";
}

TEST_F(TestNoActivation, Test_Backward_Before_Forward) {
  EXPECT_THROW(noActivation.backward(), BackwardBeforeForwardException);
}

TEST_F(TestNoActivation, Test_Equal) {
  ASSERT_EQ(noActivation, NoActivation())
      << "All no activation functions should equal.";
  ASSERT_NE(noActivation, ReLU())
      << "A no activation function is not a ReLU function.";
}
#pragma endregion NoActivation

#pragma region ReLU
class TestReLU : public ::testing::Test {
protected:
  ReLU relu;
  Eigen::MatrixXd
      X = Eigen::VectorXd::LinSpaced(20, -10, 9).reshaped(4, 5).transpose(),
      Y = Eigen::MatrixXd{{0, 0, 0, 0},
                          {0, 0, 0, 0},
                          {0, 0, 0, 1},
                          {2, 3, 4, 5},
                          {6, 7, 8, 9}},
      grad = Eigen::MatrixXd{
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}, {1, 1, 1, 1}, {1, 1, 1, 1},
      };
};

TEST_F(TestReLU, Test_Call) {
  Eigen::MatrixXd *originalX = &X;
  ASSERT_TRUE(relu(X).isApprox(Y)) << "Operator:\n" << X << "\n" << Y << "\n";
  ASSERT_FALSE(X.isApprox(Y) && originalX == &X)
      << "Operation was done inplace.\n";
}

TEST_F(TestReLU, Test_Forward) {
  Eigen::MatrixXd *originalX = &X;
  ASSERT_TRUE(relu.forward(X).isApprox(Y)) << "Forward:\n"
                                           << X << "\n"
                                           << Y << "\n";
  ASSERT_FALSE(X.isApprox(Y) && originalX == &X)
      << "Operation was done inplace.\n";
}

TEST_F(TestReLU, Test_Backward) {
  relu(X);
  ASSERT_TRUE(relu.backward().isApprox(grad)) << "Backward:\n"
                                              << X << "\n"
                                              << Y << "\n";
}

TEST_F(TestReLU, Test_Backward_Before_Forward) {
  EXPECT_THROW(relu.backward(), BackwardBeforeForwardException);
}

TEST_F(TestReLU, Test_Equal) {
  ASSERT_EQ(relu, ReLU()) << "All ReLU functions should equal.";
  ASSERT_NE(relu, NoActivation())
      << "A ReLU function is not a no activation function.";
}
#pragma endregion ReLU