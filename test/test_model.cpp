#include "cross_entropy_loss.hpp"
#include "linear.hpp"
#include "model.hpp"
#include "utils/exceptions.hpp"
#include <gtest/gtest.h>

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
#pragma endregion Fixtures

#pragma region Tests
TEST(Model, TestInitWithInvalidTotalEpochs) {
  std::vector<linear::Linear> layers = getLayers();
  loss::CrossEntropyLoss loss = getLoss();
  Model::KeywordArgs kwargs = getKwargs();
  kwargs.totalEpochs = -1;

  EXPECT_THROW(Model(layers, loss, kwargs),
               exceptions::model::InvalidTotalEpochException);
}
#pragma endregion Tests
} // namespace test_model