#pragma once
#include <Eigen/Dense>
#include <filesystem>
#include <nlohmann/json.hpp>

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

/*
  Normalise the data from the current range to the target range.
*/
Eigen::MatrixXd normalise(const Eigen::MatrixXd &data,
                          std::pair<float, float> from,
                          std::pair<float, float> to);

/*
  Flatten the matrix to be 1xN matrix.
*/
Eigen::MatrixXd flatten(const Eigen::MatrixXd &in);
#pragma endregion Matrices

#pragma region Path
/*
  Recursively find all files with matching extensions
*/
std::vector<std::filesystem::path> glob(const std::filesystem::path &path,
                                        std::vector<std::string> extensions);
#pragma endregion Path

#pragma region Image
/*
  Open the provided image path as an eigen matrix.
*/
Eigen::MatrixXd openImageAsMatrix(std::filesystem::path path);

/*
  Normalise the image data matrix from [0, 255] to [-1, 1]
*/
Eigen::MatrixXd normaliseImage(const Eigen::MatrixXd &data);
#pragma endregion Image
} // namespace utils