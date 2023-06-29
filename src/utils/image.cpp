#include "image.hpp"
#include "../exceptions/utils.hpp"
#include "math.hpp"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <utility>

Eigen::MatrixXd utils::image::openAsMatrix(std::filesystem::path path) {
  Eigen::MatrixXd result;
  cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
  if (img.empty()) {
    throw exceptions::utils::image::InvalidImageFileException(path);
  }
  cv::cv2eigen(img, result);
  return result;
}

Eigen::MatrixXd utils::image::normalise(const Eigen::MatrixXd &data) {
  return utils::math::normalise(data, std::make_pair(0, 255),
                                std::make_pair(-1, 1));
}