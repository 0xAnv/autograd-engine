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

TEST(TensorTest, ContiguousLayout) {
  // default initialized tensors are contiguous
  Tensor t({2, 3, 4}, DType::Float32);
  EXPECT_TRUE(t.is_contiguous());

  // verify strides are correct for contiguous tensor
  // For shape [2, 3, 4], strides should be [12, 4, 1]
  EXPECT_EQ(t.strides().size(), 3);
  EXPECT_EQ(t.strides()[0], 12);
  EXPECT_EQ(t.strides()[1], 4);
  EXPECT_EQ(t.strides()[2], 1);
}

TEST(TensorTest, Transpose) {
  Tensor t({2, 3}, DType::Float32);
  Tensor transposed = t.transpose(0, 1);

  // verify shape and strides
  // Original shape: [2, 3], strides: [3, 1]
  // Transposed shape: [3, 2], strides: [1, 3]
  EXPECT_EQ(transposed.shape()[0], 3);
  EXPECT_EQ(transposed.shape()[1], 2);
  EXPECT_EQ(transposed.strides()[0], 1);
  EXPECT_EQ(transposed.strides()[1], 3);

  // maintain the same underlying data pointer
  EXPECT_EQ(t.data_ptr(), transposed.data_ptr());
}

TEST(TensorTest, View) {
  Tensor t({2, 3, 4}, DType::Float32); // 24 elements
  Tensor viewed = t.view({4, 6});

  // verify shape and strides
  EXPECT_EQ(viewed.shape()[0], 4);
  EXPECT_EQ(viewed.shape()[1], 6);
  EXPECT_EQ(viewed.strides()[0], 6);
  EXPECT_EQ(viewed.strides()[1], 1);

  // view works returning a contiguous reshaped tensor
  EXPECT_TRUE(viewed.is_contiguous());

  // underlying data pointer points to the same content
  EXPECT_EQ(t.data_ptr(), viewed.data_ptr());
}