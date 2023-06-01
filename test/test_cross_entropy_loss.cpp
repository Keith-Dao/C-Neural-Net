#include "cross_entropy_loss.hpp"
#include "exceptions.hpp"
#include <gtest/gtest.h>

using namespace loss;

namespace test_loss {
#pragma region Init
TEST(CrossEntropyLoss, TestInit) {
  CrossEntropyLoss loss("mean");
  loss = CrossEntropyLoss("sum");
}

TEST(CrossEntropyLoss, TestInitWithInvalidReduction) {
  EXPECT_THROW(CrossEntropyLoss loss("INVALID"),
               src_exceptions::InvalidReductionException);
}
#pragma endregion Init

#pragma region Properties
#pragma region Reduction
TEST(CrossEntropyLoss, TestGetReduction) {
  CrossEntropyLoss loss;
  EXPECT_EQ("mean", loss.getReduction());
  loss = CrossEntropyLoss("sum");
  EXPECT_EQ("sum", loss.getReduction());
}

TEST(CrossEntropyLoss, TestSetReduction) {
  CrossEntropyLoss loss;
  loss.setReduction("sum");
  EXPECT_EQ("sum", loss.getReduction());
}

TEST(CrossEntropyLoss, TestSetReductionWithInvalidReduction) {
  CrossEntropyLoss loss;
  EXPECT_THROW(loss.setReduction("INVALID"),
               src_exceptions::InvalidReductionException);
}
#pragma endregion Reduction
#pragma endregion Properties
} // namespace test_loss