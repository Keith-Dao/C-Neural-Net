#include "cross_entropy_loss.hpp"
#include "image_loader.hpp"
#include "linear.hpp"
#include "model.hpp"
#include "utils/exceptions.hpp"
#include <Eigen/src/Core/NumTraits.h>
#include <gtest/gtest.h>
#include <memory>
#include <string>
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

  int size() const override {
    return (this->X.rows() + this->batchSize - 1) / this->batchSize;
  }

  loader::minibatch operator[](int i) const override {
    int start = i * this->batchSize,
        end =
            std::min(this->y.size(), (unsigned long)(i + 1) * this->batchSize);
    Eigen::MatrixXd X(end - start, this->X.cols());
    for (int j = start; j < end; ++j) {
      X.row(j - start) = this->X.row(j);
    }
    std::vector<int> y(this->y.begin() + start, this->y.begin() + end);
    return std::make_pair(X, y);
  };
};

struct MockLoader : public loader::ImageLoader {
  Eigen::MatrixXd trainX, valX;
  std::vector<int> trainY, valY, labels;

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
    this->labels = std::vector<int>(classes.begin(), classes.end());
    this->classes = std::vector<std::string>(this->labels.size());
    for (int i = 0; i < this->labels.size(); ++i) {
      this->classes[i] = std::to_string(this->labels[i]);
    }
  };

  std::shared_ptr<loader::DatasetBatcher>
  getBatcher(std::string type, int batchSize,
             const loader::DatasetBatcher::KeywordArgs &kwargs =
                 loader::DatasetBatcher::KeywordArgs()) const override {
    if (type != "train" && type != "test") {
      throw "Invalid type";
    }
    return std::make_shared<MockDatasetBatcher>(MockDatasetBatcher(
        type == "train" ? this->trainX : this->valX,
        type == "train" ? this->trainY : this->valY, batchSize));
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

#pragma region Train
TEST(Model, TestTrainNoValidation) {
  int epochs = 1;
  Model model = getModel();
  MockLoader loader(1);
  model.train(loader, 1e-4, 1, epochs);
  ASSERT_EQ(epochs, model.getTotalEpochs());

  // Check loss history
  std::vector<float> lossHistory{0.6935848934440013};
  ASSERT_EQ(lossHistory.size(), model.getTrainMetrics()["loss"].size())
      << "Train loss history size does not match.";
  for (int i = 0; i < lossHistory.size(); ++i) {
    ASSERT_TRUE(std::abs(lossHistory[i] -
                         std::get<float>(model.getTrainMetrics()["loss"][i])) <
                Eigen::NumTraits<float>::dummy_precision())
        << "Train loss at index " << i
        << " does not match. Expected: " << lossHistory[i]
        << ", got: " << std::get<float>(model.getTrainMetrics()["loss"][i]);
  }
  lossHistory = {};
  ASSERT_EQ(lossHistory.size(), model.getValidationMetrics()["loss"].size())
      << "Validation loss history size does not match.";

  // Check parameters
  std::vector<Eigen::MatrixXd> weights{
      Eigen::MatrixXd{
          {0.999998755067, 1.000002096106, 1.000000522309, 0.999998187911},
          {0.999998755067, 1.000002096106, 1.000000522309, 0.999998187911},
          {0.999998755067, 1.000002096106, 1.000000522309, 0.999998187911}},
      Eigen::MatrixXd{{0.998065222731, 0.998065222731, 0.998065222731},
                      {1.001934777269, 1.001934777269, 1.001934777269}}};
  std::vector<Eigen::VectorXd> biases{
      Eigen::VectorXd{{0.9999999267982, 0.9999999267982, 0.9999999267982}},
      Eigen::VectorXd{{0.999912134981, 1.000087865019}}};
  for (int i = 0; i < weights.size(); ++i) {
    ASSERT_TRUE(weights[i].isApprox(model.getLayers()[i].getWeight()))
        << "Weights of layer " << i << " does not match.";
    ASSERT_TRUE(biases[i].isApprox(model.getLayers()[i].getBias()))
        << "Bias of layer " << i << " does not match.";
  }
}

TEST(Model, TestTrainWithValidation) {
  int epochs = 1;
  Model model = getModel();
  MockLoader loader(0.7);
  model.train(loader, 1e-4, 1, epochs);
  ASSERT_EQ(epochs, model.getTotalEpochs());

  // Check loss history
  std::vector<float> lossHistory{0.7016281735019937};
  ASSERT_EQ(lossHistory.size(), model.getTrainMetrics()["loss"].size())
      << "Train loss history size does not match.";
  for (int i = 0; i < lossHistory.size(); ++i) {
    ASSERT_TRUE(std::abs(lossHistory[i] -
                         std::get<float>(model.getTrainMetrics()["loss"][i])) <
                Eigen::NumTraits<float>::dummy_precision())
        << "Train loss at index " << i
        << " does not match. Expected: " << lossHistory[i]
        << ", got: " << std::get<float>(model.getTrainMetrics()["loss"][i]);
  }
  lossHistory = {0.6748015101206345};
  ASSERT_EQ(lossHistory.size(), model.getValidationMetrics()["loss"].size())
      << "Validation loss history size does not match.";
  for (int i = 0; i < lossHistory.size(); ++i) {
    ASSERT_TRUE(
        std::abs(lossHistory[i] -
                 std::get<float>(model.getValidationMetrics()["loss"][i])) <
        Eigen::NumTraits<float>::dummy_precision())
        << "Validation loss at index " << i
        << " does not match. Expected: " << lossHistory[i] << ", got: "
        << std::get<float>(model.getValidationMetrics()["loss"][i]);
  }

  // Check parameters
  std::vector<Eigen::MatrixXd> weights{
      Eigen::MatrixXd{
          {0.999999277295, 1.000000688537, 1.00000001873, 0.999997713475},
          {0.999999277295, 1.000000688537, 1.00000001873, 0.999997713475},
          {0.999999277295, 1.000000688537, 1.00000001873, 0.999997713475}},
      Eigen::MatrixXd{{0.998819881654, 0.998819881654, 0.998819881654},
                      {1.001180118346, 1.001180118346, 1.001180118346}}};
  std::vector<Eigen::VectorXd> biases{
      Eigen::VectorXd{{1.000000008979, 1.000000008979, 1.000000008979}},
      Eigen::VectorXd{{0.999909294313, 1.000090705687}}};
  for (int i = 0; i < weights.size(); ++i) {
    ASSERT_TRUE(weights[i].isApprox(model.getLayers()[i].getWeight()))
        << "Weights of layer " << i << " does not match.";
    ASSERT_TRUE(biases[i].isApprox(model.getLayers()[i].getBias()))
        << "Bias of layer " << i << " does not match.";
  }
}

TEST(Model, TestTrainWithValidationMultipleEpoch) {
  int epochs = 3;
  Model model = getModel();
  MockLoader loader(0.7);
  model.train(loader, 1e-4, 1, epochs);
  ASSERT_EQ(epochs, model.getTotalEpochs());

  // Check loss history
  std::vector<float> lossHistory{0.7016281735019942, 0.6905228054354886,
                                 0.6832302979740018};
  ASSERT_EQ(lossHistory.size(), model.getTrainMetrics()["loss"].size())
      << "Train loss history size does not match.";
  for (int i = 0; i < lossHistory.size(); ++i) {
    ASSERT_TRUE(std::abs(lossHistory[i] -
                         std::get<float>(model.getTrainMetrics()["loss"][i])) <
                Eigen::NumTraits<float>::dummy_precision())
        << "Train loss at index " << i
        << " does not match. Expected: " << lossHistory[i]
        << ", got: " << std::get<float>(model.getTrainMetrics()["loss"][i]);
  }
  lossHistory = {0.6748015101206345, 0.6608838491713995, 0.6502008469335846};
  ASSERT_EQ(lossHistory.size(), model.getValidationMetrics()["loss"].size())
      << "Validation loss history size does not match.";
  for (int i = 0; i < lossHistory.size(); ++i) {
    ASSERT_TRUE(
        std::abs(lossHistory[i] -
                 std::get<float>(model.getValidationMetrics()["loss"][i])) <
        Eigen::NumTraits<float>::dummy_precision())
        << "Validation loss at index " << i
        << " does not match. Expected: " << lossHistory[i] << ", got: "
        << std::get<float>(model.getValidationMetrics()["loss"][i]);
  }

  // Check parameters
  std::vector<Eigen::MatrixXd> weights{
      Eigen::MatrixXd{
          {0.999999463368, 1.000004564799, 1.000004417718, 0.999989087074},
          {0.999999463368, 1.000004564799, 1.000004417718, 0.999989087074},
          {0.999999463368, 1.000004564799, 1.000004417718, 0.999989087074}},
      Eigen::MatrixXd{{0.997116223774, 0.997116223774, 0.997116223774},
                      {1.002883776226, 1.002883776226, 1.002883776226}}};
  std::vector<Eigen::VectorXd> biases{
      Eigen::VectorXd{{1.000000421266, 1.000000421266, 1.000000421266}},
      Eigen::VectorXd{{0.999763917669, 1.000236082331}}};
  for (int i = 0; i < weights.size(); ++i) {
    ASSERT_TRUE(weights[i].isApprox(model.getLayers()[i].getWeight()))
        << "Weights of layer " << i << " does not match.";
    ASSERT_TRUE(biases[i].isApprox(model.getLayers()[i].getBias()))
        << "Bias of layer " << i << " does not match.";
  }
}

TEST(Model, TestTrainWithValidationLargerBatchSize) {
  int epochs = 1;
  Model model = getModel();
  MockLoader loader(0.7);
  model.train(loader, 1e-4, 3, epochs);
  ASSERT_EQ(epochs, model.getTotalEpochs());

  // Check loss history
  std::vector<float> lossHistory{1.62205669796401};
  ASSERT_EQ(lossHistory.size(), model.getTrainMetrics()["loss"].size())
      << "Train loss history size does not match.";
  for (int i = 0; i < lossHistory.size(); ++i) {
    ASSERT_TRUE(std::abs(lossHistory[i] -
                         std::get<float>(model.getTrainMetrics()["loss"][i])) <
                Eigen::NumTraits<float>::dummy_precision())
        << "Train loss at index " << i
        << " does not match. Expected: " << lossHistory[i]
        << ", got: " << std::get<float>(model.getTrainMetrics()["loss"][i]);
  }
  lossHistory = {2.02241995571785};
  ASSERT_EQ(lossHistory.size(), model.getValidationMetrics()["loss"].size())
      << "Validation loss history size does not match.";
  for (int i = 0; i < lossHistory.size(); ++i) {
    ASSERT_TRUE(
        std::abs(lossHistory[i] -
                 std::get<float>(model.getValidationMetrics()["loss"][i])) <
        Eigen::NumTraits<float>::dummy_precision())
        << "Validation loss at index " << i
        << " does not match. Expected: " << lossHistory[i] << ", got: "
        << std::get<float>(model.getValidationMetrics()["loss"][i]);
  }

  // Check parameters

  std::vector<Eigen::MatrixXd> weights{
      Eigen::MatrixXd{
          {1.000000065579, 0.999999892849, 1.000000853661, 0.999998408936},
          {1.000000065579, 0.999999892849, 1.000000853661, 0.999998408936},
          {1.000000065579, 0.999999892849, 1.000000853661, 0.999998408936}},
      Eigen::MatrixXd{{0.998776001136, 0.998776001136, 0.998776001136},
                      {1.001223998864, 1.001223998864, 1.001223998864}}};
  std::vector<Eigen::VectorXd> biases{
      Eigen::VectorXd{{1.000000123572, 1.000000123572, 1.000000123572}},
      Eigen::VectorXd{{0.999907388883, 1.000092611117}}};
  for (int i = 0; i < weights.size(); ++i) {
    ASSERT_TRUE(weights[i].isApprox(model.getLayers()[i].getWeight()))
        << "Weights of layer " << i << " does not match.";
    ASSERT_TRUE(biases[i].isApprox(model.getLayers()[i].getBias()))
        << "Bias of layer " << i << " does not match.";
  }
}
#pragma endregion Train

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