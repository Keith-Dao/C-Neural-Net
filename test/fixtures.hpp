#pragma once
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>

namespace test_filesystem {
class FileSystemFixture : public ::testing::Test {
protected:
  std::filesystem::path root;
  std::vector<Eigen::MatrixXd> data{
      Eigen::MatrixXd{{1, 4, 5}, {2, 5, 6}, {6, 5, 6}},
      Eigen::MatrixXd{{4, 5, 1}, {5, 8, 4}, {8, 5, 9}},
      Eigen::MatrixXd{{3, 8, 9}, {6, 5, 8}, {6, 4, 1}}};
  std::vector<int> labels{0, 0, 1};

  void SetUp() override {
    std::filesystem::path tempRoot(testing::TempDir());
    std::string tempDir = std::to_string(rand());
    while (std::filesystem::exists(tempRoot / tempDir)) {
      tempDir = std::to_string(rand());
    }
    this->root = tempRoot / tempDir;
    std::filesystem::create_directory(this->root);
    std::filesystem::create_directory(this->root / "0");
    std::filesystem::create_directory(this->root / "0" / "a");
    std::filesystem::create_directory(this->root / "0" / "b");
    std::filesystem::create_directory(this->root / "1");
    std::filesystem::create_directory(this->root / "1" / "a");
    std::filesystem::create_directory(this->root / "2");
    std::filesystem::create_directory(this->root / "2" / "a");

    // Add the files
    std::vector<std::pair<std::string, std::string>> files = {
        {"0", ".png"}, {"0", ".txt"}, {"0", ".jpg"}, {"1", ".txt"},
        {"0", ".png"}, {"1", ".png"}, {"2", ".exe"}};

    for (int i = 0, j = 0; i < files.size(); ++i) {
      std::string filename = this->root / files[i].first / "a" /
                             (std::to_string(i) + files[i].second);
      if (files[i].second != ".png") {
        std::ofstream ofs(filename);
        ofs.close();
      } else {
        cv::Mat image;
        cv::eigen2cv(this->data[j++], image);
        cv::imwrite(filename, image);
      }
    }
  }

  void TearDown() override { std::filesystem::remove_all(this->root); }
};
} // namespace test_filesystem