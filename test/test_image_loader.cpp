#include "exceptions.hpp"
#include "fixtures.hpp"
#include "image_loader.hpp"
#include "utils.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace loader;

namespace test_image_loader {
using ImageLoaderFileSystem = test_filesystem::FileSystemFixture;
#pragma region Image loader
struct ImageLoaderData {
  float trainTestSplit;

  ImageLoaderData(float trainTestSplit) : trainTestSplit(trainTestSplit){};
};
std::ostream &operator<<(std::ostream &os, ImageLoaderData const &fixture) {
  return os << std::to_string(fixture.trainTestSplit);
}

class TestImageLoader : public ImageLoaderFileSystem,
                        public testing::WithParamInterface<ImageLoaderData> {};

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
struct DatasetBatcherData {
  int batchSize, size;
  bool dropLast;

  DatasetBatcherData(int batchSize, int size, bool dropLast)
      : batchSize(batchSize), size(size), dropLast(dropLast){};
};
std::ostream &operator<<(std::ostream &os, DatasetBatcherData const &fixture) {
  return os << std::to_string(fixture.batchSize) + "-" +
                   (fixture.dropLast ? "True" : "False");
}

class TestDatasetBatcher
    : public ImageLoaderFileSystem,
      public testing::WithParamInterface<DatasetBatcherData> {
public:
  DatasetBatcher getBatcher() {
    return DatasetBatcher(root, utils::glob(root, {".png"}),
                          ImageLoader::standardPreprocessing,
                          {{"0", 0}, {"1", 1}, {"2", 2}}, GetParam().batchSize,
                          false, GetParam().dropLast);
  }
};
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

#pragma region Size
TEST_P(TestDatasetBatcher, TestDatasetBatcherSize) {
  DatasetBatcher batcher = getBatcher();
  int expectedSize = GetParam().size;
  EXPECT_EQ(expectedSize, batcher.size());
}
#pragma endregion Size

#pragma region Data
INSTANTIATE_TEST_SUITE_P(, TestDatasetBatcher,
                         ::testing::Values(DatasetBatcherData(1, 3, false),
                                           DatasetBatcherData(1, 3, true),
                                           DatasetBatcherData(2, 2, false),
                                           DatasetBatcherData(2, 1, true),
                                           DatasetBatcherData(3, 1, false),
                                           DatasetBatcherData(3, 1, true),
                                           DatasetBatcherData(4, 1, false),
                                           DatasetBatcherData(4, 0, true)));
#pragma endregion Data
#pragma endregion Dataset batcher
} // namespace test_image_loader