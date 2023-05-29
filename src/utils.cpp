#include "utils.hpp"
#include "exceptions.hpp"
#include <iostream>

#pragma region Matrices
Eigen::MatrixXd utils::from_json(const json &values) {
  if (values.empty()) {
    return {};
  }

  if (!values.is_array() || !values[0].is_array()) {
    throw src_exceptions::JSONArray2DException();
  }

  int rows = values.size(), cols = values[0].size();
  Eigen::MatrixXd result(rows, cols);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (!values[i][j].is_number()) {
        throw src_exceptions::JSONTypeException();
      }
      result(i, j) = values[i][j];
    }
  }

  return result;
}
#pragma endregion Matrices