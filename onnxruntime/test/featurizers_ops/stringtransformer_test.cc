// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/StringFeaturizer.h"

namespace dft = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace test {

TEST(StringTransformer, Integer_values) {
  OpTester test("StringTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  
  // State input is needed, but no actual state is required.
  test.AddInput<uint8_t>("State", {0}, {});
  
  // We are adding a scalar Tensor in this instance
  test.AddInput<int64_t>("Input", {5}, {1, 3, 5, 7, 9});

  // Expected output.
  test.AddOutput<std::string>("Output", {5}, {"1", "3", "5", "7", "9"});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(StringTransformer, Double_values) {
  OpTester test("StringTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  
  // State input is needed, but no actual state is required.
  test.AddInput<uint8_t>("State", {0}, {});

  // We are adding a scalar Tensor in this instance
  test.AddInput<double>("Input", {5}, {1, 3, 5, 7, 9});

  // Expected output.
  test.AddOutput<std::string>("Output", {5}, {"1.000000", "3.000000", "5.000000", "7.000000", "9.000000"});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(StringTransformer, Bool_values) {
  OpTester test("StringTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // State input is needed, but no actual state is required.
  test.AddInput<uint8_t>("State", {0}, {});

  // We are adding a scalar Tensor in this instance
  test.AddInput<bool>("Input", {5}, {true, false, false, false, true});

  // Expected output.
  test.AddOutput<std::string>("Output", {5}, {"True", "False", "False", "False", "True"});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(StringTransformer, String_values) {
  OpTester test("StringTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // State input is needed, but no actual state is required.
  test.AddInput<uint8_t>("State", {0}, {});

  // We are adding a scalar Tensor in this instance
  test.AddInput<std::string>("Input", {5}, {"ONE", "three", "FIVE", "SeVeN", "NINE"});

  // Expected output.
  test.AddOutput<std::string>("Output", {5}, {"ONE", "three", "FIVE", "SeVeN", "NINE"});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

}  // namespace test
}  // namespace onnxruntime
