#pragma once
#include <Eigen/Dense>

namespace utils::math {
/*
  Convert a vector of classes to a one-hot encoded matrix.
*/
Eigen::MatrixXi oneHotEncode(const std::vector<int> &targets, int numClasses);

/*
  The softmax function.
*/
template <typename T> Eigen::MatrixXd softmax(const Eigen::MatrixBase<T> &in) {
  Eigen::MatrixXd result = in.template cast<double>();
  result.colwise() -= result.rowwise().maxCoeff();
  result = result.array().exp().matrix();
  result.array().colwise() /= result.rowwise().sum().array();
  return result;
}

/*
  The log softmax function.
*/
template <typename T>
Eigen::MatrixXd logSoftmax(const Eigen::MatrixBase<T> &in) {
  Eigen::MatrixXd result = in.template cast<double>();
  result.colwise() -= result.rowwise().maxCoeff();
  result.array().colwise() -= result.array().exp().rowwise().sum().log();
  return result;
}

/*
  Normalise the data from the current range to the target range.
*/
Eigen::MatrixXd normalise(const Eigen::MatrixXd &data,
                          std::pair<float, float> from,
                          std::pair<float, float> to);

} // namespace utils::math