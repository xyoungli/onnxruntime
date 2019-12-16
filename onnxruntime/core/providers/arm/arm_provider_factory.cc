// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/arm/arm_provider_factory.h"
#include <atomic>
#include "arm_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

struct ARMProviderFactory : IExecutionProviderFactory {
  explicit ARMProviderFactory(bool create_arena) : create_arena_(create_arena) {}
  ~ARMProviderFactory() override = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
};

std::unique_ptr<IExecutionProvider> ARMProviderFactory::CreateProvider() {
  ARMExecutionProviderInfo info;
  info.create_arena = create_arena_;
  return onnxruntime::make_unique<ARMExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ARM(int use_arena) {
  return std::make_shared<onnxruntime::ARMProviderFactory>(use_arena != 0);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_ARM, _In_ OrtSessionOptions* options, int use_arena) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_ARM(use_arena));
  return nullptr;
}
