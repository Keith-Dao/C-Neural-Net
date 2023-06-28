#include "fixtures.hpp"
#include "utils/all.hpp"
#include "utils/exceptions.hpp"
#include "utils/string.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using namespace utils;
using json = nlohmann::json;

namespace test_utils {
#pragma region Matrices
#pragma region JSON
struct MatrixData {
  Eigen::MatrixXd matrix;
  json value;

  MatrixData(const Eigen::MatrixXd &matrix, const json &value)
      : matrix(matrix), value(value){};
};
std::ostream &operator<<(std::ostream &os, MatrixData const &fixture) {
  return os << fixture.value;
}
class TestMatrixJson : public testing::TestWithParam<MatrixData> {};

TEST_P(TestMatrixJson, TestMatrixToJson) {
  auto [matrix, values] = GetParam();
  ASSERT_EQ(values, matrix::toJson(matrix));
}

TEST_P(TestMatrixJson, TestJsonToMatrix) {
  auto [matrix, values] = GetParam();
  ASSERT_TRUE(matrix.isApprox(matrix::fromJson(values)));
}

INSTANTIATE_TEST_SUITE_P(
    Utils, TestMatrixJson,
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
  EXPECT_THROW(matrix::fromJson(values), exceptions::json::JSONArray2DException)
      << "1D arrays are not supported, only 2D.";

  values = json::parse(R"([[[1, 2]]])");
  EXPECT_THROW(matrix::fromJson(values), exceptions::json::JSONArray2DException)
      << "Higher dimension arrays are not supported, only 2D.";
}

TEST(MatrixUtils, TestJsonToMatrixWithInvalidTypes) {
  json values = json::parse(R"([[true, false]])");
  EXPECT_THROW(matrix::fromJson(values), exceptions::json::JSONTypeException)
      << "Booleans are not supported, only numbers.";

  values = json::parse(R"([["a", "b"]])");
  EXPECT_THROW(matrix::fromJson(values), exceptions::json::JSONTypeException)
      << "Strings are not supported, only numbers.";

  values = json::parse(R"([[null]])");
  EXPECT_THROW(matrix::fromJson(values), exceptions::json::JSONTypeException)
      << "null is not supported, only numbers.";

  values = json::parse(R"([[{"a": "b"}, {"b": "c"}]])");
  EXPECT_THROW(matrix::fromJson(values), exceptions::json::JSONTypeException)
      << "JSON objects are not supported, only numbers.";

  values = json::parse(R"({"a": "b"})");
  EXPECT_THROW(matrix::fromJson(values), exceptions::json::JSONTypeException)
      << "JSON objects are not supported, only 2D.";
}
#pragma endregion JSON

#pragma region Flatten
TEST(MatrixUtils, TestFlatten) {
  Eigen::MatrixXd data{{1, 2, 3}}, expected{{1, 2, 3}};
  ASSERT_TRUE(expected.isApprox(matrix::flatten(data)))
      << "First flatten failed.";

  data = Eigen::MatrixXd{{1}, {2}, {3}};
  ASSERT_TRUE(expected.isApprox(matrix::flatten(data)))
      << "Second flatten failed.";

  data = Eigen::MatrixXd{{1, 2, 3, 4}, {5, 6, 7, 8}};
  expected = Eigen::MatrixXd{{1, 2, 3, 4, 5, 6, 7, 8}};
  ASSERT_TRUE(expected.isApprox(matrix::flatten(data)))
      << "Last flatten failed.";
}
#pragma endregion Flatten
#pragma endregion Matrices

#pragma region Math
#pragma region One hot encode
TEST(MathUtils, TestOneHotEncode) {
  std::vector<int> labels{0, 1, 2};
  int classes = 3;
  Eigen::MatrixXi encoded{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  ASSERT_TRUE(encoded.isApprox(math::oneHotEncode(labels, classes)))
      << "First one hot encode failed.";

  labels = std::vector<int>{1, 0, 2};
  encoded = Eigen::MatrixXi{{0, 1, 0}, {1, 0, 0}, {0, 0, 1}};
  ASSERT_TRUE(encoded.isApprox(math::oneHotEncode(labels, classes)))
      << "Second one hot encode failed.";

  labels = std::vector<int>{3};
  classes = 10;
  encoded = Eigen::MatrixXi{{0, 0, 0, 1, 0, 0, 0, 0, 0, 0}};
  ASSERT_TRUE(encoded.isApprox(math::oneHotEncode(labels, classes)))
      << "Last one hot encode failed.";
}

TEST(MathUtils, TestOneHotEncodeWithInvalidLabels) {
  std::vector<int> labels{0, 1, 3};
  int classes = 3;
  EXPECT_THROW(math::oneHotEncode(labels, classes),
               exceptions::utils::one_hot_encode::InvalidLabelIndexException);
}
#pragma endregion One hot encode

#pragma region Softmax
TEST(MathUtils, TestSoftmax) {
  Eigen::MatrixXi x{{1, 1, 1}};
  Eigen::MatrixXd trueP{{0.33333333333333, 0.33333333333333, 0.33333333333333}};
  ASSERT_TRUE(trueP.isApprox(math::softmax(x))) << "First softmax failed.";

  x = Eigen::MatrixXi{{1, 0, 0}};
  trueP = Eigen::MatrixXd{{0.576116884766, 0.211941557617, 0.211941557617}};
  ASSERT_TRUE(trueP.isApprox(math::softmax(x))) << "Second softmax failed.";

  x = Eigen::MatrixXi{{999, 0, 0}};
  trueP = Eigen::MatrixXd{{1, 0, 0}};
  ASSERT_TRUE(trueP.isApprox(math::softmax(x))) << "Last softmax failed.";
}
#pragma endregion Softmax

#pragma region Log softmax
TEST(MathUtils, TestLogSoftmax) {
  Eigen::MatrixXi x{{1, 1, 1}};
  Eigen::MatrixXd trueP{{-1.098612288668, -1.098612288668, -1.098612288668}};
  ASSERT_TRUE(trueP.isApprox(math::logSoftmax(x)))
      << "First log softmax failed.";

  x = Eigen::MatrixXi{{1, 0, 0}};
  trueP = Eigen::MatrixXd{{-0.551444713932, -1.551444713932, -1.551444713932}};
  ASSERT_TRUE(trueP.isApprox(math::logSoftmax(x)))
      << "Second log softmax failed.";

  x = Eigen::MatrixXi{{-1, -1, -1}};
  trueP = Eigen::MatrixXd{{-1.098612288668, -1.098612288668, -1.098612288668}};
  ASSERT_TRUE(trueP.isApprox(math::logSoftmax(x)))
      << "Third log softmax failed.";

  x = Eigen::MatrixXi{{999, 0, 0}};
  trueP = Eigen::MatrixXd{{0, -999, -999}};
  ASSERT_TRUE(trueP.isApprox(math::logSoftmax(x)))
      << "Last log softmax failed.";
}
#pragma endregion Log softmax

#pragma region Normalise
TEST(MathUtils, TestNormalise) {
  Eigen::MatrixXd data{{0, 127.5, 255}}, expected{{0, 0.5, 1}};
  std::pair<float, float> from{0, 255}, to{0, 1};
  EXPECT_TRUE(expected.isApprox(math::normalise(data, from, to)))
      << "First normalise failed.";

  expected << -1, 0, 1;
  to = std::make_pair(-1, 1);
  EXPECT_TRUE(expected.isApprox(math::normalise(data, from, to)))
      << "Second normalise failed.";

  expected << -2, 0, 2;
  to = std::make_pair(-2, 2);
  EXPECT_TRUE(expected.isApprox(math::normalise(data, from, to)))
      << "Third normalise failed.";

  expected << -2, 0.5, 3;
  to = std::make_pair(-2, 3);
  EXPECT_TRUE(expected.isApprox(math::normalise(data, from, to)))
      << "Last normalise failed.";
}

TEST(MathUtils, TestNormaliseWithInvalidRange) {
  Eigen::MatrixXd data{{0, 127.5, 255}};
  std::pair<float, float> from{0, -1}, to{0, 1};
  EXPECT_THROW(math::normalise(data, from, to),
               exceptions::utils::normalise::InvalidRangeException)
      << "First normalise did not throw.";

  std::swap(from, to);
  EXPECT_THROW(math::normalise(data, from, to),
               exceptions::utils::normalise::InvalidRangeException)
      << "Second normalise did not throw.";
}
#pragma endregion Normalise

#pragma region Logits to prediction
TEST(MathUtils, TestLogitsToPrediction) {
  std::vector<Eigen::MatrixXd> logits{Eigen::MatrixXd{{1, 2, 3}},
                                      Eigen::MatrixXd{{5, 2, 1}, {2, 24, 1}}};
  std::vector<std::vector<int>> expected{{2}, {0, 1}};
  for (int i = 0; i < logits.size(); ++i) {
    std::vector<int> predictions = math::logitsToPrediction(logits[i]);
    ASSERT_EQ(expected[i].size(), predictions.size())
        << "Sizes did not match on test " << i;
    ASSERT_EQ(expected[i], predictions) << "Values did not match on test " << i;
  }
}
#pragma endregion Logits to prediction
#pragma endregion Math

#pragma region Path
#pragma region Glob
using UtilsGlob = test_filesystem::FileSystemWithImageDataFixture;
TEST_F(UtilsGlob, TestGlob) {
  std::vector<std::filesystem::path> expected = {root / "0" / "a" / "0.png",
                                                 root / "0" / "a" / "4.png",
                                                 root / "1" / "a" / "5.png"};
  std::vector<std::filesystem::path> result = path::glob(root, {".png"});
  std::sort(result.begin(), result.end());
  ASSERT_EQ(expected, result) << "Glob .png only";

  expected = {root / "0" / "a" / "0.png", root / "0" / "a" / "1.txt",
              root / "0" / "a" / "2.jpg", root / "0" / "a" / "4.png",
              root / "1" / "a" / "3.txt", root / "1" / "a" / "5.png"};
  result = path::glob(root, {".png", ".jpg", ".txt"});
  std::sort(result.begin(), result.end());
  ASSERT_EQ(expected, result) << "Glob .png, .jpg and .txt.";
}
#pragma endregion Glob
#pragma endregion Path

#pragma region Image
using TestImageUtils = test_filesystem::FileSystemWithImageDataFixture;
#pragma region Open image
TEST_F(TestImageUtils, TestOpenImageAsMatrix) {
  ASSERT_TRUE(data[0].isApprox(image::openAsMatrix(root / "0" / "a" / "0.png")))
      << "Opened image data is not equivalent.";
}

TEST_F(TestImageUtils, TestOpenImageAsMatrixWithNonImage) {
  EXPECT_THROW(image::openAsMatrix(root / "0" / "a" / "1.txt"),
               exceptions::utils::image::InvalidImageFileException)
      << "Cannot open non image file as a matrix.";
}
#pragma endregion Open image

#pragma region Normalise
TEST(ImageUtils, TestNormaliseImage) {
  Eigen::MatrixXd data{{0, 127.5, 255}}, expected{{-1, 0, 1}};
  ASSERT_TRUE(expected.isApprox(image::normalise(data)))
      << "Normalise image failed.";
}
#pragma endregion Normalise
#pragma endregion Image

#pragma region String
TEST(StringUtils, TestSplit) {
  std::vector<std::string> strings{"abc 123 bd", "abc. 123. bd"},
      delimiters{" ", ". "}, expected{"abc", "123", "bd"};
  for (int i = 0; i < strings.size(); ++i) {
    EXPECT_EQ(string::split(strings[i], delimiters[i]), expected)
        << "Split failed on test " << i;
  }
}

TEST(StringUtils, TestJoin) {
  std::vector<std::string> expected{"abc 123 bd", "abc. 123. bd"},
      delimiters{" ", ". "}, strings{"abc", "123", "bd"};
  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(string::join(strings, delimiters[i]), expected[i])
        << "Join failed on test " << i;
  }
  EXPECT_EQ(string::join({}, " "), "")
      << "Join failed when no strings are provided.";
}

TEST(StringUtils, TestFloatToNum) {
  std::string expected = "1.234";
  EXPECT_EQ(expected, string::floatToString(1.2341, 3));
  EXPECT_EQ(expected, string::floatToString(1.2335, 3));

  double x = 1.15478;
  expected = "1.154780";
  EXPECT_EQ(expected, string::floatToString(x, 6));
}

TEST(StringUtils, TestCapitalise) {
  std::string expected = "ABC";
  std::vector<std::string> inputs{"ABC", "aBC"};
  for (const std::string &input : inputs) {
    EXPECT_EQ(expected, string::capitalise(input))
        << input << " failed to capitalise.";
  }
}
#pragma endregion String
} // namespace test_utils