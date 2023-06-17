#pragma once
#include <Eigen/Dense>
#include <filesystem>

namespace utils::image {
/*
  Open the provided image path as an eigen matrix.
*/
Eigen::MatrixXd openAsMatrix(std::filesystem::path path);

/*
  Normalise the image data matrix from [0, 255] to [-1, 1]
*/
Eigen::MatrixXd normalise(const Eigen::MatrixXd &data);
} // namespace utils::image