#pragma once
#include <Eigen/Dense>

namespace utils {
#pragma region Matrices
/*
  Convert a matrix to its stringified form.
*/
template <typename T>
std::string to_string(const Eigen::MatrixBase<T> &matrix) {
  std::string result = "[";

  for (int i = 0; i < matrix.rows(); ++i) {
    if (i != 0) {
      result += ",";
    }

    std::string row = "[";
    for (int j = 0; j < matrix.cols(); ++j) {
      if (j != 0) {
        row += ",";
      }
      row += std::to_string(matrix(i, j));
    }
    result += row + "]";
  }

  return result + "]";
};
#pragma endregion Matrices
} // namespace utils