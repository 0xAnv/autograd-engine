#include "ag/tensor.hpp"
#include <gtest/gtest.h>

TEST(TensorTest, ConstructorAndSize) {
  // create a 2 x 3 float 32 Tensor
  std::vector<int> shape = {2, 3};
  Tensor t(shape, DType::Float32);

  // verify shape and elements
  EXPECT_EQ(t.shape().size(), 2);
  EXPECT_EQ(t.shape()[0], 2);
  EXPECT_EQ(t.shape()[1], 3);
  EXPECT_EQ(t.numel(), 6);

  // verify total memory allocations size
  // For 6 Float32 elements, we expect 6 * 4 = 24 bytes
  EXPECT_EQ(t.numel() * sizeof_dtype(t.dtype()), 24);
}