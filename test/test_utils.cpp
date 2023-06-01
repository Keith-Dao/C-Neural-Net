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

#pragma region One hot encode
TEST(MatrixUtils, TestOneHotEncode) {
  std::vector<int> labels{0, 1, 2};
  int classes = 3;
  Eigen::MatrixXi encoded{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  ASSERT_TRUE(encoded.isApprox(one_hot_encode(labels, classes)))
      << "First one hot encode failed.";

  labels = std::vector<int>{1, 0, 2};
  encoded = Eigen::MatrixXi{{0, 1, 0}, {1, 0, 0}, {0, 0, 1}};
  ASSERT_TRUE(encoded.isApprox(one_hot_encode(labels, classes)))
      << "Second one hot encode failed.";

  labels = std::vector<int>{3};
  classes = 10;
  encoded = Eigen::MatrixXi{{0, 0, 0, 1, 0, 0, 0, 0, 0, 0}};
  ASSERT_TRUE(encoded.isApprox(one_hot_encode(labels, classes)))
      << "Last one hot encode failed.";
}

TEST(MatrixUtils, TestOneHotEncodeWithInvalidLabels) {
  std::vector<int> labels{0, 1, 3};
  int classes = 3;
  EXPECT_THROW(one_hot_encode(labels, classes),
               src_exceptions::InvalidLabelIndexException);
}
#pragma endregion One hot encode

#pragma region Softmax
TEST(MatrixUtils, TestSoftmax) {
  Eigen::MatrixXd x{{1, 1, 1}},
      trueP{{0.33333333333333, 0.33333333333333, 0.33333333333333}};
  ASSERT_TRUE(trueP.isApprox(softmax(x))) << "First softmax failed.";

  x = Eigen::MatrixXd{{1, 0, 0}};
  trueP = Eigen::MatrixXd{{0.576116884766, 0.211941557617, 0.211941557617}};
  ASSERT_TRUE(trueP.isApprox(softmax(x))) << "Second softmax failed.";

  x = Eigen::MatrixXd{{999, 0, 0}};
  trueP = Eigen::MatrixXd{{1, 0, 0}};
  ASSERT_TRUE(trueP.isApprox(softmax(x))) << "Last softmax failed.";
}
#pragma endregion Softmax

#pragma region Log softmax
TEST(MatrixUtils, TestLogSoftmax) {
  Eigen::MatrixXd x{{1, 1, 1}},
      trueP{{-1.098612288668, -1.098612288668, -1.098612288668}};
  ASSERT_TRUE(trueP.isApprox(log_softmax(x))) << "First log softmax failed.";

  x = Eigen::MatrixXd{{1, 0, 0}};
  trueP = Eigen::MatrixXd{{-0.551444713932, -1.551444713932, -1.551444713932}};
  ASSERT_TRUE(trueP.isApprox(log_softmax(x))) << "Second log softmax failed.";

  x = Eigen::MatrixXd{{-1, -1, -1}};
  trueP = Eigen::MatrixXd{{-1.098612288668, -1.098612288668, -1.098612288668}};
  ASSERT_TRUE(trueP.isApprox(log_softmax(x))) << "Third log softmax failed.";

  x = Eigen::MatrixXd{{999, 0, 0}};
  trueP = Eigen::MatrixXd{{0, -999, -999}};
  ASSERT_TRUE(trueP.isApprox(log_softmax(x))) << "Last log softmax failed.";
}
#pragma endregion Log softmax
#pragma endregion Matrices
} // namespace test_utils