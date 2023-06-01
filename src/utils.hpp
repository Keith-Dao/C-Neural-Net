#pragma once
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace utils {
#pragma region Matrices
/*
  Convert a matrix to a nested json array.
*/
template <typename T> json to_json(const Eigen::MatrixBase<T> &matrix) {
  json result = json::array();

  for (int i = 0; i < matrix.rows(); ++i) {
    json row;
    for (int j = 0; j < matrix.cols(); ++j) {
      row.push_back(matrix(i, j));
    }
    result.push_back(row);
  }

  return result;
};

/*
  Convert a nested json array to a matrix.
*/
Eigen::MatrixXd from_json(const json &values);

/*
  Convert a vector of classes to a one-hot encoded matrix.
*/
Eigen::MatrixXi one_hot_encode(const std::vector<int> &targets, int numClasses);

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
Eigen::MatrixXd log_softmax(const Eigen::MatrixBase<T> &in) {
  Eigen::MatrixXd result = in.template cast<double>();
  result.colwise() -= result.rowwise().maxCoeff();
  result.array().colwise() -= result.array().exp().rowwise().sum().log();
  return result;
}
#pragma endregion Matrices
} // namespace utils