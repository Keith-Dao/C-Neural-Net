#pragma once
#include <Eigen/Dense>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <unordered_set>

using json = nlohmann::json;

namespace utils {
#pragma region Matrices
/*
  Convert a matrix to a nested json array.
*/
template <typename T> json toJson(const Eigen::MatrixBase<T> &matrix) {
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
Eigen::MatrixXd fromJson(const json &values);

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
#pragma endregion Matrices

#pragma region Path
/*
  Recursively find all files with matching extensions
*/
std::vector<std::filesystem::path> glob(const std::filesystem::path &path,
                                        std::vector<std::string> extensions);
#pragma endregion Path
} // namespace utils