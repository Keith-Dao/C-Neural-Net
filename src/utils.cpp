#include "utils.hpp"
#include "exceptions.hpp"
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <utility>

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

Eigen::MatrixXd utils::normalise(const Eigen::MatrixXd &data,
                                 std::pair<float, float> from,
                                 std::pair<float, float> to) {
  auto [fromMin, fromMax] = from;
  if (fromMin >= fromMax) {
    throw exceptions::utils::normalise::InvalidRangeException();
  }

  auto [toMin, toMax] = to;
  if (toMin >= toMax) {
    throw exceptions::utils::normalise::InvalidRangeException();
  }

  return (data.array() - fromMin) * (toMax - toMin) / (fromMax - fromMin) +
         toMin;
}
#pragma endregion Matrices

#pragma region Path
std::vector<std::filesystem::path>
utils::glob(const std::filesystem::path &path,
            std::vector<std::string> extensions) {
  std::unordered_set<std::string> extensionSet(extensions.begin(),
                                               extensions.end());
  std::vector<std::filesystem::path> result;
  for (auto const &entry :
       std::filesystem::recursive_directory_iterator(path)) {
    if (!std::filesystem::is_directory(entry.path()) &&
        extensionSet.count(entry.path().extension())) {
      result.push_back(entry.path());
    }
  };
  return result;
};
#pragma endregion Path

#pragma region Image
Eigen::MatrixXd utils::openImageAsMatrix(std::filesystem::path path) {
  Eigen::MatrixXd result;
  cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
  if (img.empty()) {
    throw exceptions::utils::image::InvalidImageFileException(path);
  }
  cv::cv2eigen(img, result);
  return result;
}

Eigen::MatrixXd utils::normaliseImage(const Eigen::MatrixXd &data) {
  return utils::normalise(data, std::make_pair(0, 255), std::make_pair(-1, 1));
}
#pragma endregion Image