#include "cross_entropy_loss.hpp"
#include "exceptions.hpp"
#include <gtest/gtest.h>
#include <tuple>

using namespace loss;

namespace test_loss {
#pragma region Fixture
enum DataSize { smallClose, smallExact, smallFar, large };
std::string getDataSizeName(DataSize dataSize) {
  switch (dataSize) {
  case smallClose:
    return "small close";
  case smallExact:
    return "small exact";
  case smallFar:
    return "small far";
  case large:
    return "Large";
  default:
    throw "Invalid";
  }
}

#pragma region Data
typedef std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, std::vector<int>> data;
data getSmallClose() {
  Eigen::MatrixXd logits{{1, 0, 0}};
  Eigen::MatrixXi encoded{{1, 0, 0}};
  std::vector<int> labels{0};
  return std::make_tuple(logits, encoded, labels);
}

data getSmallExact() {
  Eigen::MatrixXd logits{{0, 999, 0}};
  Eigen::MatrixXi encoded{{0, 1, 0}};
  std::vector<int> labels{1};
  return std::make_tuple(logits, encoded, labels);
}

data getSmallFar() {
  Eigen::MatrixXd logits{{0, 1, 1}};
  Eigen::MatrixXi encoded{{1, 0, 0}};
  std::vector<int> labels{0};
  return std::make_tuple(logits, encoded, labels);
}

data getLarge() {
  Eigen::MatrixXd logits{{1, 0, 0}, {999, 0, 0}, {0, 1, 1}};
  Eigen::MatrixXi encoded{{1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
  std::vector<int> labels{0, 0, 0};
  return std::make_tuple(logits, encoded, labels);
}

data getData(DataSize dataSize) {
  switch (dataSize) {
  case smallClose:
    return getSmallClose();
  case smallExact:
    return getSmallExact();
  case smallFar:
    return getSmallFar();
  case large:
    return getLarge();
  default:
    throw "Invalid";
  }
}
#pragma endregion Data

#pragma region Values
double getSmallCloseLoss(std::string reduction) { return 0.5514447139320511; }

double getSmallExactLoss(std::string reduction) { return 0; }

double getSmallFarLoss(std::string reduction) { return 1.8619948040582512; }

double getLargeLoss(std::string reduction) {
  if (reduction == "sum") {
    return 2.4134395179903025;
  }
  if (reduction == "mean") {
    return 0.8044798393301008;
  }
  throw "Invalid";
}

double getLoss(std::string reduction, DataSize dataSize) {
  switch (dataSize) {
  case smallClose:
    return getSmallCloseLoss(reduction);
  case smallExact:
    return getSmallExactLoss(reduction);
  case smallFar:
    return getSmallFarLoss(reduction);
  case large:
    return getLargeLoss(reduction);
  default:
    throw "Invalid";
  }
}
#pragma endregion Values
struct FixtureData {
  std::string reduction;
  DataSize dataSize;

  FixtureData(std::string reduction, DataSize(dataSize))
      : reduction(reduction), dataSize(dataSize){};
};
std::ostream &operator<<(std::ostream &os, FixtureData const &fixture) {
  return os << fixture.reduction + " - " + getDataSizeName(fixture.dataSize);
}

class TestCrossEntropyLoss : public testing::TestWithParam<FixtureData> {};
#pragma endregion Fixture

#pragma region Init
TEST(CrossEntropyLoss, TestInit) {
  CrossEntropyLoss loss("mean");
  loss = CrossEntropyLoss("sum");
}

TEST(CrossEntropyLoss, TestInitWithInvalidReduction) {
  EXPECT_THROW(CrossEntropyLoss loss("INVALID"),
               src_exceptions::InvalidReductionException);
}
#pragma endregion Init

#pragma region Properties
#pragma region Reduction
TEST(CrossEntropyLoss, TestGetReduction) {
  CrossEntropyLoss loss;
  EXPECT_EQ("mean", loss.getReduction());
  loss = CrossEntropyLoss("sum");
  EXPECT_EQ("sum", loss.getReduction());
}

TEST(CrossEntropyLoss, TestSetReduction) {
  CrossEntropyLoss loss;
  loss.setReduction("sum");
  EXPECT_EQ("sum", loss.getReduction());
}

TEST(CrossEntropyLoss, TestSetReductionWithInvalidReduction) {
  CrossEntropyLoss loss;
  EXPECT_THROW(loss.setReduction("INVALID"),
               src_exceptions::InvalidReductionException);
}
#pragma endregion Reduction
#pragma endregion Properties

#pragma region Forward
TEST_P(TestCrossEntropyLoss, TestForward) {
  CrossEntropyLoss loss(GetParam().reduction);
  auto [logits, oneHot, labels] = getData(GetParam().dataSize);
  double lossValue = getLoss(GetParam().reduction, GetParam().dataSize);
  ASSERT_EQ(lossValue, loss.forward(logits, oneHot))
      << "Forward with one hot encoded.";
  ASSERT_EQ(lossValue, loss.forward(logits, labels)) << "Forward with labels.";
}

TEST(CrossEntropyLoss, TestForwardWithMissingValues) {
  CrossEntropyLoss loss;
  Eigen::MatrixXd logits{{1, 1, 1}};
  std::vector<int> labels;
  EXPECT_THROW(loss.forward(logits, labels),
               src_exceptions::EmptyMatrixException)
      << "Missing labels.";

  Eigen::MatrixXi oneHot;
  EXPECT_THROW(loss.forward(logits, oneHot),
               src_exceptions::EmptyMatrixException)
      << "Missing one hot encoded labels.";

  oneHot = Eigen::MatrixXi{{1, 0, 0}};
  logits = Eigen::MatrixXd();
  EXPECT_THROW(loss.forward(logits, oneHot),
               src_exceptions::EmptyMatrixException)
      << "Missing logits.";

  oneHot = Eigen::MatrixXi();
  EXPECT_THROW(loss.forward(logits, oneHot),
               src_exceptions::EmptyMatrixException)
      << "Logits and one hot encoding labels are empty.";
}

TEST(CrossEntropyLoss, TestForwardWithMismatchedShapes) {
  CrossEntropyLoss loss;
  Eigen::MatrixXd logits{{1, 1, 1}};
  std::vector<int> labels{2, 1};
  EXPECT_THROW(loss.forward(logits, labels),
               src_exceptions::InvalidShapeException)
      << "Extra labels.";

  Eigen::MatrixXi oneHot{{0, 0, 1}, {0, 1, 0}};
  EXPECT_THROW(loss.forward(logits, oneHot),
               src_exceptions::InvalidShapeException)
      << "Extra one hot encoded labels.";

  logits = Eigen::MatrixXd{{1, 1, 1}, {1, 1, 1}};
  labels = std::vector<int>{1};
  EXPECT_THROW(loss.forward(logits, labels),
               src_exceptions::InvalidShapeException)
      << "Extra logits on label call.";

  oneHot = Eigen::MatrixXi{{0, 1, 0}};
  EXPECT_THROW(loss.forward(logits, labels),
               src_exceptions::InvalidShapeException)
      << "Extra logits on one hot encoded call.";

  oneHot = Eigen::MatrixXi{{0, 1}, {0, 1}};
  EXPECT_THROW(loss.forward(logits, labels),
               src_exceptions::InvalidShapeException)
      << "Shape mismatch.";
}

TEST(CrossEntropyLoss, TestForwardWithInvalidLabel) {
  CrossEntropyLoss loss;
  Eigen::MatrixXd logits{{1, 1, 1}};
  std::vector<int> labels{3};
  EXPECT_THROW(loss.forward(logits, labels),
               src_exceptions::InvalidLabelIndexException);
}
#pragma endregion Forward

#pragma region Builtins
TEST_P(TestCrossEntropyLoss, TestCall) {
  CrossEntropyLoss loss(GetParam().reduction);
  auto [logits, oneHot, labels] = getData(GetParam().dataSize);
  double lossValue = getLoss(GetParam().reduction, GetParam().dataSize);
  ASSERT_EQ(lossValue, loss(logits, oneHot)) << "Call with one hot encoded.";
  ASSERT_EQ(lossValue, loss(logits, labels)) << "Call with labels.";
}
#pragma endregion Builtins

#pragma region Data
INSTANTIATE_TEST_SUITE_P(, TestCrossEntropyLoss,
                         ::testing::Values(FixtureData("mean", smallClose),
                                           FixtureData("sum", smallClose),
                                           FixtureData("mean", smallExact),
                                           FixtureData("sum", smallExact),
                                           FixtureData("mean", smallFar),
                                           FixtureData("sum", smallFar),
                                           FixtureData("mean", large),
                                           FixtureData("sum", large),
                                           FixtureData("sum", large)));
#pragma endregion Data
} // namespace test_loss