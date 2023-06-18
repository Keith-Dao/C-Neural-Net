#include "metrics.hpp"
#include "utils/exceptions.hpp"
#include "utils/matrix.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/NumTraits.h>
#include <gtest/gtest.h>
#include <limits>

using namespace metrics;

#pragma region Confusion matrix
TEST(Metrics, TestNewConfusionMatrix) {
  Eigen::MatrixXi expected{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  ASSERT_TRUE(expected.isApprox(getNewConfusionMatrix(3)))
      << "Did not get a 3x3 matrix of zeros.";
}

TEST(Metrics, TestNewConfusionMatrixWithInvalidValue) {
  EXPECT_THROW(getNewConfusionMatrix(-1),
               exceptions::metrics::InvalidNumberOfClassesException)
      << "Exception did not throw when number of classes are negative.";

  EXPECT_THROW(getNewConfusionMatrix(0),
               exceptions::metrics::InvalidNumberOfClassesException)
      << "Exception did not throw when number of classes is zero.";
}

struct ConfusionMatrixData {
  Eigen::MatrixXi confusionMatrix, expected;
  std::vector<int> predicted, actual;

  ConfusionMatrixData(const Eigen::MatrixXi &confusionMatrix,
                      const Eigen::MatrixXi &expected,
                      const std::vector<int> &predicted,
                      const std::vector<int> &actual)
      : confusionMatrix(confusionMatrix), expected(expected),
        predicted(predicted), actual(actual){};
};
std::ostream &operator<<(std::ostream &os, ConfusionMatrixData const &fixture) {
  return os << utils::matrix::toJson(fixture.confusionMatrix) << " -> "
            << utils::matrix::toJson(fixture.expected);
}
class TestConfusionMatrix : public testing::TestWithParam<ConfusionMatrixData> {
};

TEST_P(TestConfusionMatrix, TestAddToConfusionMatrix) {
  Eigen::MatrixXi matrix = GetParam().confusionMatrix;
  addToConfusionMatrix(matrix, GetParam().predicted, GetParam().actual);
  ASSERT_TRUE(GetParam().expected.isApprox(matrix));
}

INSTANTIATE_TEST_SUITE_P(
    Metrics, TestConfusionMatrix,
    ::testing::Values(
        ConfusionMatrixData(Eigen::MatrixXi{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
                            Eigen::MatrixXi{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                            {0, 1, 2}, {0, 1, 2}),
        ConfusionMatrixData(Eigen::MatrixXi{{0, 1, 1}, {1, 0, 0}, {0, 0, 0}},
                            Eigen::MatrixXi{{1, 1, 2}, {2, 0, 0}, {0, 1, 0}},
                            {1, 2, 0, 0}, {0, 1, 2, 0})));

TEST(Metrics, TestAddToConfusionMatrixWithInvalidInputs) {
  std::vector<int> predicted{0, 1, 2}, actual{1};
  Eigen::MatrixXi matrix{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  EXPECT_THROW(addToConfusionMatrix(matrix, predicted, actual),
               exceptions::metrics::InvalidDatasetException)
      << "Did not throw exception when size of predicted and actual is "
         "mismatched.";
}
#pragma endregion Confusion matrix

#pragma region Metrics
#pragma region Accuracy
TEST(Metrics, TestAccuracy) {
  std::vector<Eigen::MatrixXi> confusionMatrices{
      Eigen::MatrixXi{{5, 0, 0}, {0, 2, 0}, {0, 0, 10}},
      Eigen::MatrixXi{{4, 4, 2}, {0, 2, 0}, {3, 2, 5}},
      Eigen::MatrixXi{{20, 1, 60}, {29, 13, 2}, {32, 6, 34}}};
  std::vector<float> expected{1, 0.5, 0.340101522843};

  for (int i = 0; i < confusionMatrices.size(); ++i) {
    ASSERT_TRUE(std::abs(expected[i] - accuracy(confusionMatrices[i])) <
                Eigen::NumTraits<float>::dummy_precision())
        << "Test " << i << " failed.";
  }
}
#pragma endregion Accuracy

#pragma region Precision
TEST(Metrics, TestPrecision) {
  std::vector<Eigen::MatrixXi> confusionMatrices{
      Eigen::MatrixXi{{3, 2}, {1, 4}},
      Eigen::MatrixXi{{4, 4, 2}, {0, 2, 0}, {3, 2, 5}},
      Eigen::MatrixXi{{1, 0, 0}, {0, 0, 0}, {0, 0, 0}},
      Eigen::MatrixXi{{20, 1, 60}, {29, 13, 2}, {32, 6, 34}},
      Eigen::MatrixXi{
          {50, 3, 0, 0}, {26, 8, 0, 1}, {20, 2, 4, 0}, {12, 0, 0, 1}}};
  std::vector<std::vector<float>> expected{
      {0.6, 0.8},
      {0.4, 1, 0.5},
      {1, 0, 0},
      {0.24691358024691357, 0.29545454545454547, 0.4722222222222222},
      {0.9433962264150944, 0.22857142857142856, 0.15384615384615385,
       0.07692307692307693}};

  for (int i = 0; i < confusionMatrices.size(); ++i) {
    std::vector<float> result = precision(confusionMatrices[i]);
    ASSERT_EQ(expected[i].size(), result.size())
        << "Precision result does not return the same size as expected, got:"
        << result.size() << ", expected: " << expected[i].size();

    for (int j = 0; j < expected[i].size(); ++j) {
      ASSERT_TRUE(std::abs(expected[i][j] - result[j]) <
                  Eigen::NumTraits<float>::dummy_precision())
          << "Test " << i << " index " << j << " failed.";
    }
  }
}
#pragma endregion Precision
#pragma endregion Metrics