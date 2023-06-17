#include "math.hpp"
#include "exceptions.hpp"

Eigen::MatrixXi utils::math::oneHotEncode(const std::vector<int> &targets,
                                          int numClasses) {
  Eigen::MatrixXi result = Eigen::MatrixXi::Zero(targets.size(), numClasses);
  for (int i = 0; i < targets.size(); ++i) {
    if (targets[i] >= numClasses || targets[i] < 0) {
      throw exceptions::utils::one_hot_encode::InvalidLabelIndexException();
    }
    result(i, targets[i]) = 1;
  }
  return result;
}

Eigen::MatrixXd utils::math::normalise(const Eigen::MatrixXd &data,
                                       std::pair<float, float> from,
                                       std::pair<float, float> to) {
  auto [fromMin, fromMax] = from;
  if (fromMin >= fromMax) {
    throw exceptions::utils::normalise::InvalidRangeException();
  }

  auto [toMin, toMax] = to;
  if (toMin >= toMax) {
    throw exceptions::utils::normalise::InvalidRangeException();
  }

  return (data.array() - fromMin) * (toMax - toMin) / (fromMax - fromMin) +
         toMin;
}
