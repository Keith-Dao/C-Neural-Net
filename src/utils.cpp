#include "utils.hpp"
#include "exceptions.hpp"
#include <filesystem>

#pragma region Matrices
Eigen::MatrixXd utils::fromJson(const json &values) {
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

Eigen::MatrixXi utils::oneHotEncode(const std::vector<int> &targets,
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
#pragma endregion Matrices

#pragma region Path
std::vector<std::string>
utils::glob(const std::filesystem::path &path,
            std::unordered_set<std::string> extensions) {
  std::vector<std::string> result;
  for (auto const &entry :
       std::filesystem::recursive_directory_iterator(path)) {
    if (!std::filesystem::is_directory(entry.path()) &&
        extensions.count(entry.path().extension())) {
      result.push_back(entry.path());
    }
  };
  return result;
};
#pragma endregion Path