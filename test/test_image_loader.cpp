#include "exceptions.hpp"
#include "fixtures.hpp"
#include "image_loader.hpp"
#include "utils.hpp"
#include <gtest/gtest.h>

using namespace loader;

#define TRAIN_TEST_SPLIT 0
#define BATCH_SIZE 1

namespace test_image_loader {
using ImageLoaderFileSystem = test_filesystem::FileSystemFixture;
#pragma region Image loader
class TestImageLoader
    : public ImageLoaderFileSystem,
      public testing::WithParamInterface<std::tuple<float, int>> {
public:
  ImageLoader getImageLoader() {
    return ImageLoader(root, {utils::flatten}, {".png"},
                       std::get<TRAIN_TEST_SPLIT>(GetParam()), false);
  }

  static ImageLoader getImageLoader(std::filesystem::path root,
                                    float trainTestSplit) {
    return ImageLoader(root, {utils::flatten}, {".png"}, trainTestSplit, false);
  }
};

#pragma region Init
TEST_P(TestImageLoader, TestImageLoaderInit) {
  ImageLoader loader = getImageLoader();
  ASSERT_EQ((int)(std::get<TRAIN_TEST_SPLIT>(GetParam()) * 3),
            loader.getTrainFiles().size())
      << "Mismatched train size";
  ASSERT_EQ(3 - (int)(std::get<TRAIN_TEST_SPLIT>(GetParam()) * 3),
            loader.getTestFiles().size())
      << "Mismatched test size";
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

#pragma region Properties
#pragma region Classes
TEST_F(ImageLoaderFileSystem, TestImageLoaderClasses) {
  ImageLoader loader(root, ImageLoader::standardPreprocessing, {".png"}, 0.7,
                     false);
  ASSERT_EQ(classes, loader.getClasses()) << "Classes did not match.";

  std::vector<std::string> alteredClasses = loader.getClasses();
  alteredClasses[0] = "ALTERED";
  ASSERT_NE(alteredClasses, loader.getClasses())
      << "Class getter returns a reference  to the member.";
}
#pragma endregion Classes
#pragma endregion Properties

#pragma region Batcher
TEST_P(TestImageLoader, TestImageLoaderGetBatcher) {
  ImageLoader loader = getImageLoader();
  DatasetBatcher::KeywordArgs kwargs;
  kwargs.shuffle = false;
  int batchSize = std::get<BATCH_SIZE>(GetParam());

  // Test the train and test dataset batcher
  std::pair<std::string, std::vector<std::filesystem::path>> datasets[2] = {
      std::make_pair("train", loader.getTrainFiles()),
      std::make_pair("test", loader.getTestFiles())};
  for (auto [datasetName, datasetFiles] : datasets) {
    DatasetBatcher datasetBatcher =
        loader.getBatcher(datasetName, batchSize, kwargs);
    for (int i = 0; i < (datasetFiles.size() + batchSize - 1) / batchSize;
         ++i) {
      auto [result, resultLabels] = datasetBatcher[i];

      // Find the order in which the files have been loaded.
      std::vector<int> indexes;
      for (int j = 0; j < batchSize && i * batchSize + j < datasetFiles.size();
           ++j) {
        indexes.push_back(dataIndex[datasetFiles[i * batchSize + j]]);
      }

      // Check labels
      std::vector<int> expectedLabels(indexes.size());
      for (int j = 0; j < indexes.size(); ++j) {
        expectedLabels[j] = labels[indexes[j]];
      }
      ASSERT_EQ(expectedLabels, resultLabels)
          << "Labels do not match on batch " << i << " of " << datasetName
          << std::endl;

      // Construct the expected matrix
      if (indexes.size() <= 0) {
        throw "Test error: Missing data rows.";
      }
      Eigen::MatrixXd expected(indexes.size(), data[0].size());
      for (int j = 0; j < indexes.size(); ++j) {
        expected.row(j) = utils::flatten(data[indexes[j]]);
      }
      ASSERT_EQ(expected.rows(), result.rows())
          << "Number of rows don't match on batch " << i << " for "
          << datasetName << std::endl
          << "Expected: " << expected.rows() << ", Got: " << result.rows();
      ASSERT_EQ(expected.cols(), result.cols())
          << "Number of cols don't match on batch " << i << " for "
          << datasetName << std::endl
          << "Expected: " << expected.cols() << ", Got: " << result.cols();
      ASSERT_TRUE(expected.isApprox(result))
          << "Data does not match on batch " << i << " for " << datasetName
          << std::endl
          << "Expected: " << expected << ", Got: " << result;
    }
  }
}

TEST_F(ImageLoaderFileSystem, TestImageLoaderGetBatcherWithInvalidDataset) {
  ImageLoader loader = TestImageLoader::getImageLoader(root, 0.7);
  EXPECT_THROW(loader.getBatcher("INVALID", 0),
               exceptions::loader::InvalidDatasetException);
}
#pragma endregion Batcher

#pragma region Data
INSTANTIATE_TEST_SUITE_P(
    , TestImageLoader,
    ::testing::Combine(
        ::testing::ConvertGenerator<float>(::testing::Values(0, (float)1 / 3,
                                                             (float)2 / 3, 1)),
        ::testing::ConvertGenerator<int>(::testing::Values(1, 2, 3))));
#pragma endregion Data
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
    std::vector<std::filesystem::path> files = utils::glob(root, {".png"});
    std::sort(files.begin(), files.end());

    DatasetBatcher::KeywordArgs kwargs;
    kwargs.shuffle = false;
    kwargs.dropLast = GetParam().dropLast;
    return DatasetBatcher(root, files, {utils::flatten},
                          {{"0", 0}, {"1", 1}, {"2", 2}}, GetParam().batchSize,
                          kwargs);
  }
};

#pragma region Init
TEST_F(ImageLoaderFileSystem, TestDatasetBatcherInit) {
  DatasetBatcher(
      root, utils::glob(root, {".png"}), ImageLoader::standardPreprocessing,
      {{"0", 0}, {"1", 1}, {"2", 2}}, 1, DatasetBatcher::KeywordArgs());
}

TEST_F(ImageLoaderFileSystem,
       TestDatasetBatcherInitWithoutExplicitKeywordArgs) {
  DatasetBatcher(root, utils::glob(root, {".png"}),
                 ImageLoader::standardPreprocessing,
                 {{"0", 0}, {"1", 1}, {"2", 2}}, 1);
}
TEST_F(ImageLoaderFileSystem, TestDatasetBatcherInitWithInvalidBatchSize) {
  EXPECT_THROW(DatasetBatcher(root, utils::glob(root, {".png"}),
                              ImageLoader::standardPreprocessing,
                              {{"0", 0}, {"1", 1}, {"2", 2}}, 0,
                              DatasetBatcher::KeywordArgs()),
               exceptions::loader::InvalidBatchSizeException);
}
#pragma endregion Init

#pragma region Index
TEST_P(TestDatasetBatcher, TestDatasetBatcherIndex) {
  DatasetBatcher batcher = getBatcher();
  for (int i = 0, j = 0; i < GetParam().size; ++i) {
    auto [result, resultLabels] = batcher[i];

    std::vector<int> expectedLabels(
        labels.begin() + i * GetParam().batchSize,
        std::min(labels.end(),
                 labels.begin() + (i + 1) * GetParam().batchSize));
    ASSERT_EQ(expectedLabels, resultLabels)
        << "Labels do not match on batch " << i << std::endl;
    // Construct the expected matrix
    int rows = std::min((unsigned long)GetParam().batchSize,
                        data.size() - i * GetParam().batchSize);
    if (rows <= 0) {
      throw "Test error: Missing data rows.";
    }
    Eigen::MatrixXd expected(rows, data[0].size());
    for (int j = 0; j < rows; ++j) {
      expected.row(j) = utils::flatten(data[i * GetParam().batchSize + j]);
    }
    ASSERT_EQ(expected.rows(), result.rows())
        << "Number of rows don't match on batch " << i << std::endl
        << "Expected: " << expected.rows() << ", Got: " << result.rows();
    ASSERT_EQ(expected.cols(), result.cols())
        << "Number of cols don't match on batch " << i << std::endl
        << "Expected: " << expected.cols() << ", Got: " << result.cols();
    ASSERT_TRUE(expected.isApprox(result))
        << "Data does not match on batch " << i << std::endl
        << "Expected: " << expected << ", Got: " << result;
  }
}

TEST_P(TestDatasetBatcher, TestDatasetBatcherIndexOutOfRange) {
  DatasetBatcher batcher = getBatcher();
  EXPECT_THROW(batcher[GetParam().size], std::out_of_range)
      << "Did not throw out of range.";

  EXPECT_THROW(batcher[-1], std::out_of_range)
      << "Did not throw out of range for negative numbers.";
}
#pragma endregion Index

#pragma region Range based for loop
TEST_P(TestDatasetBatcher, TestDatasetBatcherRangeBasedForLoop) {
  DatasetBatcher batcher = getBatcher();
  int i = 0;
  for (auto &[result, resultLabels] : batcher) {
    std::vector<int> expectedLabels(
        labels.begin() + i * GetParam().batchSize,
        std::min(labels.end(),
                 labels.begin() + (i + 1) * GetParam().batchSize));
    ASSERT_EQ(expectedLabels, resultLabels)
        << "Labels do not match on batch " << i << std::endl;
    // Construct the expected matrix
    int rows = std::min((unsigned long)GetParam().batchSize,
                        data.size() - i * GetParam().batchSize);
    if (rows <= 0) {
      throw "Test error: Missing data rows.";
    }
    Eigen::MatrixXd expected(rows, data[0].size());
    for (int j = 0; j < rows; ++j) {
      expected.row(j) = utils::flatten(data[i * GetParam().batchSize + j]);
    }
    ASSERT_EQ(expected.rows(), result.rows())
        << "Number of rows don't match on batch " << i << std::endl
        << "Expected: " << expected.rows() << ", Got: " << result.rows();
    ASSERT_EQ(expected.cols(), result.cols())
        << "Number of cols don't match on batch " << i << std::endl
        << "Expected: " << expected.cols() << ", Got: " << result.cols();
    ASSERT_TRUE(expected.isApprox(result))
        << "Data does not match on batch " << i << std::endl
        << "Expected: " << expected << ", Got: " << result;
    i++;
  }
}
#pragma endregion Range based for loop

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