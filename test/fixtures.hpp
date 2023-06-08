#pragma once
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace test_filesystem {
class FileSystemFixture : public ::testing::Test {
protected:
  std::filesystem::path root;

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
    for (int i = 0; i < files.size(); ++i) {
      std::ofstream ofs(this->root / files[i].first / "a" /
                        (std::to_string(i) + files[i].second));
      ofs.close();
    }
  }

  void TearDown() override { std::filesystem::remove_all(this->root); }
};
} // namespace test_filesystem