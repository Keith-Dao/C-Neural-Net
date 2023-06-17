#include "matrix.hpp"
#include "exceptions.hpp"

Eigen::MatrixXd utils::matrix::fromJson(const json &values) {
  if (values.empty()) {
    return {};
  }

  if (!values.is_array()) {
    throw exceptions::json::JSONTypeException();
  }
  if (!values[0].is_array()) {
    throw exceptions::json::JSONArray2DException();
  }

  int rows = values.size(), cols = values[0].size();
  Eigen::MatrixXd result(rows, cols);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (values[i][j].is_array()) {
        throw exceptions::json::JSONArray2DException();
      }
      if (!values[i][j].is_number()) {
        throw exceptions::json::JSONTypeException();
      }
      result(i, j) = values[i][j];
    }
  }

  return result;
}

Eigen::MatrixXd utils::matrix::flatten(const Eigen::MatrixXd &in) {
  Eigen::MatrixXd out(in);
  return out.reshaped<Eigen::RowMajor>().transpose();
}