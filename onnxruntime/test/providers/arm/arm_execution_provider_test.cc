// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/arm/arm_execution_provider.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
TEST(ARMExecutionProviderTest, MetadataTest) {
  ARMExecutionProviderInfo info;
  auto provider = onnxruntime::make_unique<ARMExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);
  ASSERT_STREQ(provider->GetAllocator(0, OrtMemTypeDefault)->Info().name, ARM);
}
}  // namespace test
}  // namespace onnxruntime
