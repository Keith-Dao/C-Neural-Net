#include "exceptions.hpp"
#include "fixtures.hpp"
#include "image_loader.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace loader;

namespace test_image_loader {
using ImageLoaderFileSystem = test_filesystem::FileSystemFixture;
struct FixtureData {
  float trainTestSplit;

  FixtureData(float trainTestSplit) : trainTestSplit(trainTestSplit){};
};
std::ostream &operator<<(std::ostream &os, FixtureData const &fixture) {
  return os << std::to_string(fixture.trainTestSplit);
}
class TestImageLoader : public ImageLoaderFileSystem,
                        public testing::WithParamInterface<FixtureData> {};

#pragma region Init
TEST_F(ImageLoaderFileSystem, TestInit) {
  ImageLoader(root, ImageLoader::standardPreprocessing, {".png"}, 0.7, false);
}

TEST_F(ImageLoaderFileSystem, TestInitWithInvalidSplit) {
  float split = -0.1;
  EXPECT_THROW(ImageLoader(root, ImageLoader::standardPreprocessing, {".png"},
                           split, false),
               exceptions::loader::InvalidTrainTestSplitException)
      << "First invalid range did not throw.";

  split = 1.1;
  EXPECT_THROW(ImageLoader(root, ImageLoader::standardPreprocessing, {".png"},
                           split, false),
               exceptions::loader::InvalidTrainTestSplitException)
      << "Last invalid range did not throw.";
}

TEST_F(ImageLoaderFileSystem, TestImageLoaderInitWithNoFiles) {
  EXPECT_THROW(ImageLoader(root / "0" / "b", ImageLoader::standardPreprocessing,
                           {".png"}, 0.7, false),
               exceptions::loader::NoFilesFoundException)
      << "Invalid directory did not throw.";

  EXPECT_THROW(ImageLoader(root, ImageLoader::standardPreprocessing, {".gif"},
                           0.7, false),
               exceptions::loader::NoFilesFoundException)
      << "No files with a given extension did not throw.";
}
#pragma endregion Init
} // namespace test_image_loader