#pragma once
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace utils::matrix {
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
  Flatten the matrix to be 1xN matrix.
*/
Eigen::MatrixXd flatten(const Eigen::MatrixXd &in);
} // namespace utils::matrix