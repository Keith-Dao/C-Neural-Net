#include "exceptions.hpp"
#include "utils.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace utils;
using json = nlohmann::json;

namespace test_utils {
#pragma region Matrices
struct MatrixData {
  Eigen::MatrixXd matrix;
  json value;

  MatrixData(const Eigen::MatrixXd &matrix, const json &value)
      : matrix(matrix), value(value){};
};
std::ostream &operator<<(std::ostream &os, MatrixData const &fixture) {
  return os << fixture.value;
}
class TestMatrix : public testing::TestWithParam<MatrixData> {};

TEST_P(TestMatrix, TestMatrixToJson) {
  auto [matrix, values] = GetParam();
  ASSERT_EQ(values, to_json(matrix));
}

TEST_P(TestMatrix, TestJsonToMatrix) {
  auto [matrix, values] = GetParam();
  ASSERT_TRUE(matrix.isApprox(from_json(values)));
}

INSTANTIATE_TEST_SUITE_P(
    Utils, TestMatrix,
    ::testing::Values(MatrixData(Eigen::MatrixXd{{1, 2, 3}, {3, 2, 1}},
                                 json{{1, 2, 3}, {3, 2, 1}}),
                      MatrixData(Eigen::MatrixXd::Ones(3, 3),
                                 json{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}),
                      MatrixData(Eigen::VectorXd::LinSpaced(3, 1, 3),
                                 json{{1}, {2}, {3}}),
                      MatrixData(Eigen::VectorXd::LinSpaced(5, 1, 3),
                                 json{{1}, {1.5}, {2}, {2.5}, {3}}),
                      MatrixData({}, json::array())));

TEST(MatrixUtils, TestJsonToMatrixWithInvalidValues) {
  json values = json::parse(R"([1, 2])");
  EXPECT_THROW(from_json(values), src_exceptions::JSONArray2DException)
      << "1D arrays are not supported, only 2D.";

  values = json::parse(R"([[[1, 2]]])");
  EXPECT_THROW(from_json(values), src_exceptions::JSONArray2DException)
      << "Higher dimension arrays are not supported, only 2D.";
}

TEST(MatrixUtils, TestJsonToMatrixWithInvalidTypes) {
  json values = json::parse(R"([[true, false]])");
  EXPECT_THROW(from_json(values), src_exceptions::JSONTypeException)
      << "Booleans are not supported, only numbers.";

  values = json::parse(R"([["a", "b"]])");
  EXPECT_THROW(from_json(values), src_exceptions::JSONTypeException)
      << "Strings are not supported, only numbers.";

  values = json::parse(R"([[null]])");
  EXPECT_THROW(from_json(values), src_exceptions::JSONTypeException)
      << "null is not supported, only numbers.";

  values = json::parse(R"([[{"a": "b"}, {"b": "c"}]])");
  EXPECT_THROW(from_json(values), src_exceptions::JSONTypeException)
      << "JSON objects are not supported, only numbers.";

  values = json::parse(R"({"a": "b"})");
  EXPECT_THROW(from_json(values), src_exceptions::JSONTypeException)
      << "JSON objects are not supported, only 2D.";
}
#pragma endregion Matrices
} // namespace test_utils