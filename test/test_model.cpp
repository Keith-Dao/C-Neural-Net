#include "cross_entropy_loss.hpp"
#include "image_loader.hpp"
#include "linear.hpp"
#include "model.hpp"
#include "utils/exceptions.hpp"
#include <gtest/gtest.h>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace model;

namespace test_model {
#pragma region Fixtures
std::vector<linear::Linear> getLayers() {
  std::vector<linear::Linear> layers = {linear::Linear(4, 3),
                                        linear::Linear(3, 2, "ReLU")};
  for (linear::Linear &layer : layers) {
    layer.setWeight(Eigen::MatrixXd::Ones(layer.outChannels, layer.inChannels));
    layer.setBias(Eigen::VectorXd::Ones(layer.outChannels));
  }
  return layers;
}

loss::CrossEntropyLoss getLoss() { return loss::CrossEntropyLoss("sum"); }

Model::KeywordArgs getKwargs() {
  Model::KeywordArgs kwargs;
  kwargs.setTrainMetricsFromMetricTypes({"loss"});
  kwargs.setValidationMetricsFromMetricTypes({"loss"});
  kwargs.classes = {"0", "1"};
  return kwargs;
}

Model getModel() { return Model(getLayers(), getLoss(), getKwargs()); }

std::pair<Eigen::MatrixXd, std::vector<int>> getData() {
  return std::make_pair(Eigen::MatrixXd{{4, -3, 2, 4},
                                        {6, -3, 6, 1},
                                        {5, 9, 8, 3},
                                        {8, -10, 8, -7},
                                        {0, 3, 7, 5},
                                        {7, -6, 8, 8},
                                        {1, -10, -7, 5},
                                        {-6, 5, 4, -9},
                                        {4, -3, 8, 6},
                                        {5, -9, 2, 1}},
                        std::vector<int>{0, 1, 1, 1, 1, 0, 1, 1, 1, 0});
}

struct MockDatasetBatcher : public loader::DatasetBatcher {
  Eigen::MatrixXd X;
  std::vector<int> y;
  int batchSize;

  MockDatasetBatcher(const Eigen::MatrixXd &X, const std::vector<int> &y,
                     int batchSize)
      : X(X), y(y), batchSize(batchSize), loader::DatasetBatcher("", {}, {}, {},
                                                                 batchSize){};

  int getSize() {
    return (this->X.size() + this->batchSize - 1) / this->batchSize;
  }

  loader::minibatch operator[](int i) const {
    int end = std::min(y.size(), (unsigned long)i + this->batchSize);
    Eigen::MatrixXd X(end - i, this->X.cols());
    for (int j = i; j < end; ++j) {
      X.row(j - i) = this->X.row(j);
    }
    std::vector<int> y(this->y.begin() + i * this->batchSize,
                       this->y.begin() + end);
    return std::make_pair(X, y);
  };
};

struct MockLoader {
  Eigen::MatrixXd trainX, valX;
  std::vector<int> trainY, valY, classes;

  MockLoader(float split) {
    auto [X, y] = getData();
    int trainSize = X.rows() * split;
    this->trainX = Eigen::MatrixXd(trainSize, X.cols());
    for (int i = 0; i < trainSize; ++i) {
      this->trainX.row(i) = X.row(i);
    }
    this->valX = Eigen::MatrixXd(X.rows() - trainSize, X.cols());
    for (int i = trainSize; i < X.rows(); ++i) {
      this->valX.row(i - trainSize) = X.row(i);
    }

    this->trainY = std::vector<int>(y.begin(), y.begin() + trainSize);
    this->valY = std::vector<int>(y.begin() + trainSize, y.end());

    std::unordered_set<int> classes(y.begin(), y.end());
    this->classes = std::vector<int>(classes.begin(), classes.end());
  };

  MockDatasetBatcher operator()(std::string type, int batchSize) {
    if (type == "train") {
      return MockDatasetBatcher(this->trainX, this->trainY, batchSize);
    }
    if (type == "test") {
      return MockDatasetBatcher(this->valX, this->valY, batchSize);
    }
    throw "Invalid type";
  }
};

#pragma endregion Fixtures

#pragma region Tests
#pragma region Init
TEST(Model, TestInitWithInvalidTotalEpochs) {
  std::vector<linear::Linear> layers = getLayers();
  loss::CrossEntropyLoss loss = getLoss();
  Model::KeywordArgs kwargs = getKwargs();
  kwargs.totalEpochs = -1;

  EXPECT_THROW(Model(layers, loss, kwargs),
               exceptions::model::InvalidTotalEpochException);
}
#pragma endregion Init

#pragma region Properties
#pragma region Evaluation mode
TEST(Model, TestEvaluationMode) {
  Model model = getModel();
  auto checkAll = [&](bool mode) {
    ASSERT_EQ(model.getEval(), mode)
        << "Expected model's evaluation mode to be set to "
        << (mode ? "true" : "false");
    for (const auto &layer : model.getLayers()) {
      ASSERT_EQ(layer.getEval(), mode)
          << "Expected all layers' evaluation mode to be set to "
          << (mode ? "true" : "false");
    }
  };

  checkAll(false);
  model.setEval(false);
  checkAll(false);
  model.setEval(true);
  checkAll(true);
}
#pragma endregion Evaluation mode

#pragma region Layers
TEST(Model, TestLayers) {
  std::vector<std::vector<linear::Linear>> layers{
      {linear::Linear(1, 2), linear::Linear(2, 3), linear::Linear(3, 5)},
      {linear::Linear(1, 2)}};
  Model model = getModel();
  for (int i = 0; i < layers.size(); ++i) {
    model.setLayers(layers[i]);
    for (int j = 0; j < layers[i].size(); ++j) {
      ASSERT_EQ(layers[i][j], model.getLayers()[j])
          << "Layers did not match on test " << i << " layer " << j;
    }
  }
}

TEST(Model, TestLayersWithEmptyVector) {
  std::vector<linear::Linear> layers;
  Model model = getModel();
  EXPECT_THROW(model.setLayers(layers),
               exceptions::model::EmptyLayersVectorException);
}
#pragma endregion Layers

#pragma region Total epochs
TEST(Model, TestTotalEpochs) {
  Model model = getModel();
  ASSERT_EQ(0, model.getTotalEpochs()) << "Total epochs should default to 0.";

  model.setTotalEpochs(10);
  ASSERT_EQ(10, model.getTotalEpochs())
      << "Total epochs should have been changed to 10.";
}

TEST(Model, TestTotalEpochsWithInvalidValue) {
  Model model = getModel();
  EXPECT_THROW(model.setTotalEpochs(-1),
               exceptions::model::InvalidTotalEpochException)
      << "A negative value should throw the exception.";
}
#pragma endregion Total epochs

#pragma region Loss
TEST(Model, TestLoss) {
  Model model = getModel();
  loss::CrossEntropyLoss loss("mean");

  ASSERT_NE(model.getLoss(), loss)
      << "Loss should have different reduction methods.";
  model.setLoss(loss);
  ASSERT_EQ(model.getLoss(), loss)
      << "Loss should have the same reduction methods.";
}
#pragma endregion Loss
#pragma endregion Properties

#pragma region Forward pass
TEST(Model, TestForward) {
  Model model = getModel();
  Eigen::MatrixXd x = getData().first,
                  expected{{25., 25.}, {34., 34.}, {79., 79.}, {1., 1.},
                           {49., 49.}, {55., 55.}, {0., 0.},   {0., 0.},
                           {49., 49.}, {1., 1.}},
                  result = model.forward(x);

  ASSERT_TRUE(expected.isApprox(result)) << "Expected:\n"
                                         << expected << "\nGot:\n"
                                         << result;
}

TEST(Model, TestPredict) {
  Model model = getModel();
  Eigen::MatrixXd x{{0, 0, 0, 0}};
  std::vector<std::string> expected{"0"};
  ASSERT_EQ(expected, model.predict(x));
}

TEST(Model, TestPredictWithNodeClasses) {
  Model model(getLayers(), getLoss());
  Eigen::MatrixXd x{{0, 0, 0, 0}};
  EXPECT_THROW(model.predict(x), exceptions::model::MissingClassesException);
}
#pragma endregion Forward pass

#pragma region Test
TEST(Model, TestTestWithNoClasses) {
  Model model(getLayers(), getLoss());
  MockLoader loader(1);
  EXPECT_THROW(model.test(loader("train", 1)),
               exceptions::model::MissingClassesException);
}
#pragma endregion Test
#pragma endregion Tests
} // namespace test_model