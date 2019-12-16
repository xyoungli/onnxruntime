// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/arm/arm_provider_factory.h"
#include <atomic>
#include "arm_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

struct ARMProviderFactory : IExecutionProviderFactory {
  explicit ARMProviderFactory(bool create_arena, PowerMode mode=PowerMode::ARM_POWER_NO_BIND, int threads=1)
        : create_arena_(create_arena), mode_(mode), threads_(threads) {}
  ~ARMProviderFactory() override = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
  PowerMode mode_{PowerMode::ARM_POWER_NO_BIND};
  int threads_{1};
};

std::unique_ptr<IExecutionProvider> ARMProviderFactory::CreateProvider() {
  ARMExecutionProviderInfo info;
  info.create_arena = create_arena_;
  info.mode = mode_;
  info.threads = threads_;
  return onnxruntime::make_unique<ARMExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ARM(int use_arena,
        PowerMode mode=PowerMode::ARM_POWER_NO_BIND, int threads=1) {
  return std::make_shared<onnxruntime::ARMProviderFactory>(use_arena != 0, mode, threads);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_ARM, _In_ OrtSessionOptions* options, int use_arena) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_ARM(use_arena));
  return nullptr;
}
