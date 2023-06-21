#include "cross_entropy_loss.hpp"
#include "linear.hpp"
#include "model.hpp"
#include <gtest/gtest.h>

using namespace model;

namespace test_model {
#pragma region Tests
TEST(Model, TestInit) {
  std::vector<linear::Linear> layers{linear::Linear(4, 3),
                                     linear::Linear(3, 2)};
  loss::CrossEntropyLoss loss;
  Model model(layers, loss);

  Model::KeywordArgs kwargs;
  kwargs.setTrainMetricsFromMetricTypes({"loss", "accuracy"});
  model = Model(layers, loss, kwargs);
}
#pragma endregion Tests
} // namespace test_model