#include "exceptions.hpp"
#include "fixtures.hpp"
#include "utils.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

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
  ASSERT_EQ(values, toJson(matrix));
}

TEST_P(TestMatrix, TestJsonToMatrix) {
  auto [matrix, values] = GetParam();
  ASSERT_TRUE(matrix.isApprox(fromJson(values)));
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
  EXPECT_THROW(fromJson(values), exceptions::json::JSONArray2DException)
      << "1D arrays are not supported, only 2D.";

  values = json::parse(R"([[[1, 2]]])");
  EXPECT_THROW(fromJson(values), exceptions::json::JSONArray2DException)
      << "Higher dimension arrays are not supported, only 2D.";
}

TEST(MatrixUtils, TestJsonToMatrixWithInvalidTypes) {
  json values = json::parse(R"([[true, false]])");
  EXPECT_THROW(fromJson(values), exceptions::json::JSONTypeException)
      << "Booleans are not supported, only numbers.";

  values = json::parse(R"([["a", "b"]])");
  EXPECT_THROW(fromJson(values), exceptions::json::JSONTypeException)
      << "Strings are not supported, only numbers.";

  values = json::parse(R"([[null]])");
  EXPECT_THROW(fromJson(values), exceptions::json::JSONTypeException)
      << "null is not supported, only numbers.";

  values = json::parse(R"([[{"a": "b"}, {"b": "c"}]])");
  EXPECT_THROW(fromJson(values), exceptions::json::JSONTypeException)
      << "JSON objects are not supported, only numbers.";

  values = json::parse(R"({"a": "b"})");
  EXPECT_THROW(fromJson(values), exceptions::json::JSONTypeException)
      << "JSON objects are not supported, only 2D.";
}

#pragma region One hot encode
TEST(MatrixUtils, TestOneHotEncode) {
  std::vector<int> labels{0, 1, 2};
  int classes = 3;
  Eigen::MatrixXi encoded{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  ASSERT_TRUE(encoded.isApprox(oneHotEncode(labels, classes)))
      << "First one hot encode failed.";

  labels = std::vector<int>{1, 0, 2};
  encoded = Eigen::MatrixXi{{0, 1, 0}, {1, 0, 0}, {0, 0, 1}};
  ASSERT_TRUE(encoded.isApprox(oneHotEncode(labels, classes)))
      << "Second one hot encode failed.";

  labels = std::vector<int>{3};
  classes = 10;
  encoded = Eigen::MatrixXi{{0, 0, 0, 1, 0, 0, 0, 0, 0, 0}};
  ASSERT_TRUE(encoded.isApprox(oneHotEncode(labels, classes)))
      << "Last one hot encode failed.";
}

TEST(MatrixUtils, TestOneHotEncodeWithInvalidLabels) {
  std::vector<int> labels{0, 1, 3};
  int classes = 3;
  EXPECT_THROW(oneHotEncode(labels, classes),
               exceptions::utils::one_hot_encode::InvalidLabelIndexException);
}
#pragma endregion One hot encode

#pragma region Softmax
TEST(MatrixUtils, TestSoftmax) {
  Eigen::MatrixXi x{{1, 1, 1}};
  Eigen::MatrixXd trueP{{0.33333333333333, 0.33333333333333, 0.33333333333333}};
  ASSERT_TRUE(trueP.isApprox(softmax(x))) << "First softmax failed.";

  x = Eigen::MatrixXi{{1, 0, 0}};
  trueP = Eigen::MatrixXd{{0.576116884766, 0.211941557617, 0.211941557617}};
  ASSERT_TRUE(trueP.isApprox(softmax(x))) << "Second softmax failed.";

  x = Eigen::MatrixXi{{999, 0, 0}};
  trueP = Eigen::MatrixXd{{1, 0, 0}};
  ASSERT_TRUE(trueP.isApprox(softmax(x))) << "Last softmax failed.";
}
#pragma endregion Softmax

#pragma region Log softmax
TEST(MatrixUtils, TestLogSoftmax) {
  Eigen::MatrixXi x{{1, 1, 1}};
  Eigen::MatrixXd trueP{{-1.098612288668, -1.098612288668, -1.098612288668}};
  ASSERT_TRUE(trueP.isApprox(logSoftmax(x))) << "First log softmax failed.";

  x = Eigen::MatrixXi{{1, 0, 0}};
  trueP = Eigen::MatrixXd{{-0.551444713932, -1.551444713932, -1.551444713932}};
  ASSERT_TRUE(trueP.isApprox(logSoftmax(x))) << "Second log softmax failed.";

  x = Eigen::MatrixXi{{-1, -1, -1}};
  trueP = Eigen::MatrixXd{{-1.098612288668, -1.098612288668, -1.098612288668}};
  ASSERT_TRUE(trueP.isApprox(logSoftmax(x))) << "Third log softmax failed.";

  x = Eigen::MatrixXi{{999, 0, 0}};
  trueP = Eigen::MatrixXd{{0, -999, -999}};
  ASSERT_TRUE(trueP.isApprox(logSoftmax(x))) << "Last log softmax failed.";
}
#pragma endregion Log softmax
#pragma endregion Matrices

#pragma region Path
#pragma region Glob
using UtilsGlob = test_filesystem::FileSystemFixture;
TEST_F(UtilsGlob, TestGlob) {
  std::vector<std::filesystem::path> expected = {
      root / "0" / "a" / "0.png", root / "0" / "a" / "2.png",
      root / "0" / "a" / "5.png", root / "1" / "a" / "6.png"};
  std::vector<std::filesystem::path> result = glob(root, {".png"});
  std::sort(result.begin(), result.end());
  ASSERT_EQ(expected, result) << "Glob .png only";

  expected = {root / "0" / "a" / "0.png", root / "0" / "a" / "1.txt",
              root / "0" / "a" / "2.png", root / "0" / "a" / "3.jpg",
              root / "0" / "a" / "5.png", root / "1" / "a" / "4.txt",
              root / "1" / "a" / "6.png"};
  result = glob(root, {".png", ".jpg", ".txt"});
  std::sort(result.begin(), result.end());
  ASSERT_EQ(expected, result) << "Glob .png, .jpg and .txt.";
}
#pragma endregion Glob
#pragma endregion Path

#pragma region Image
class Image : public ::testing::Test {
protected:
  std::filesystem::path root;
  Eigen::MatrixXd data;

  void SetUp() override {
    std::filesystem::path tempRoot(testing::TempDir());
    std::string tempDir = std::to_string(rand());
    while (std::filesystem::exists(tempRoot / tempDir)) {
      tempDir = std::to_string(rand());
    }
    this->root = tempRoot / tempDir;
    std::filesystem::create_directory(this->root);

    this->data = Eigen::VectorXd::LinSpaced(100, 0, 99).reshaped(10, 10);
    cv::Mat image;
    cv::eigen2cv(this->data, image);
    cv::imwrite(this->root / "test.png", image);

    std::ofstream file(this->root / "test.txt");
    file << "Some data";
    file.close();
  }

  void TearDown() override { std::filesystem::remove_all(this->root); }
};

TEST_F(Image, TestOpenImageAsMatrix) {
  ASSERT_TRUE(data.isApprox(utils::openImageAsMatrix(root / "test.png")))
      << "Opened image data is not equivalent.";
}

TEST_F(Image, TestOpenImageAsMatrixWithNonImage) {
  EXPECT_THROW(utils::openImageAsMatrix(root / "test.txt"),
               exceptions::utils::image::InvalidImageFileException)
      << "Cannot open non image file as a matrix.";
}
#pragma endregion Image
} // namespace test_utils