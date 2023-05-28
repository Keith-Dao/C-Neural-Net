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
  json result;

  for (int i = 0; i < matrix.rows(); ++i) {
    json row;
    for (int j = 0; j < matrix.cols(); ++j) {
      row.push_back(matrix(i, j));
    }
    result.push_back(row);
  }

  return result;
};
#pragma endregion Matrices
} // namespace utils