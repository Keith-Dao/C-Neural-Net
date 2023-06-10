#include "exceptions.hpp"
#include "fixtures.hpp"
#include "image_loader.hpp"
#include "utils.hpp"
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

#pragma region Image loader
class TestImageLoader : public ImageLoaderFileSystem,
                        public testing::WithParamInterface<FixtureData> {};

#pragma region Init
TEST_F(ImageLoaderFileSystem, TestImageLoaderInit) {
  ImageLoader(root, ImageLoader::standardPreprocessing, {".png"}, 0.7, false);
}

TEST_F(ImageLoaderFileSystem, TestImageLoaderInitWithInvalidSplit) {
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
#pragma endregion Image loader

#pragma region Dataset batcher
#pragma region Init
TEST_F(ImageLoaderFileSystem, TestDatasetBatcherInit) {
  DatasetBatcher(root, utils::glob(root, {".png"}),
                 ImageLoader::standardPreprocessing,
                 {{"0", 0}, {"1", 1}, {"2", 2}}, 1, false);
}

TEST_F(ImageLoaderFileSystem, TestDatasetBatcherInitWithInvalidBatchSize) {
  EXPECT_THROW(DatasetBatcher(root, utils::glob(root, {".png"}),
                              ImageLoader::standardPreprocessing,
                              {{"0", 0}, {"1", 1}, {"2", 2}}, 0, false),
               exceptions::loader::InvalidBatchSizeException);
}
#pragma endregion Init
#pragma endregion Dataset batcher
} // namespace test_image_loader