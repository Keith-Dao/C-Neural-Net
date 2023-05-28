#include "utils.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>

using namespace utils;

namespace test_utils {
#pragma region Matrices
struct MatrixData {
  Eigen::MatrixXd matrix;
  std::string string;

  MatrixData(const Eigen::MatrixXd &matrix, const std::string &string)
      : matrix(matrix), string(string){};
};
std::ostream &operator<<(std::ostream &os, MatrixData const &fixture) {
  return os << fixture.string;
}
class TestMatrix : public testing::TestWithParam<MatrixData> {};

TEST_P(TestMatrix, TestMatrixToString) {
  auto [matrix, string] = GetParam();
  ASSERT_EQ(string, to_string(matrix));
}

INSTANTIATE_TEST_SUITE_P(
    Utils, TestMatrix,
    ::testing::Values(
        MatrixData(
            Eigen::MatrixXd{{1, 2, 3}, {3, 2, 1}},
            "[[1.000000,2.000000,3.000000],[3.000000,2.000000,1.000000]]"),
        MatrixData(
            Eigen::MatrixXd::Ones(3, 3),
            "[[1.000000,1.000000,1.000000],[1.000000,1.000000,1.000000],[1."
            "000000,1.000000,1.000000]]"),
        MatrixData(Eigen::VectorXd::LinSpaced(3, 1, 3),
                   "[[1.000000],[2.000000],[3.000000]]")));
#pragma endregion Matrices
} // namespace test_utils