#include "utils.hpp"
#include "exceptions.hpp"

#pragma region Matrices
Eigen::MatrixXd utils::from_json(const json &values) {
  if (values.empty()) {
    return {};
  }

  if (!values.is_array()) {
    throw src_exceptions::JSONTypeException();
  }
  if (!values[0].is_array()) {
    throw src_exceptions::JSONArray2DException();
  }

  int rows = values.size(), cols = values[0].size();
  Eigen::MatrixXd result(rows, cols);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (values[i][j].is_array()) {
        throw src_exceptions::JSONArray2DException();
      }
      if (!values[i][j].is_number()) {
        throw src_exceptions::JSONTypeException();
      }
      result(i, j) = values[i][j];
    }
  }

  return result;
}

Eigen::MatrixXi utils::one_hot_encode(const std::vector<int> &targets,
                                      int numClasses) {
  Eigen::MatrixXi result = Eigen::MatrixXi::Zero(targets.size(), numClasses);
  for (int i = 0; i < targets.size(); ++i) {
    if (targets[i] >= numClasses || targets[i] < 0) {
      throw src_exceptions::InvalidLabelIndexException();
    }
    result(i, targets[i]) = 1;
  }
  return result;
}

#pragma endregion Matrices