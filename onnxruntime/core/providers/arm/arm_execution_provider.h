// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {
// Information needed to construct ARM execution providers.
struct ARMExecutionProviderInfo {
  bool create_arena{true};

  explicit ARMExecutionProviderInfo(bool use_arena)
          : create_arena(use_arena) {}

  ARMExecutionProviderInfo() = default;
};

// Logical device representation.
class ARMExecutionProvider : public IExecutionProvider {
public:
  explicit ARMExecutionProvider(const ARMExecutionProviderInfo& info);
  virtual ~ARMExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
          const onnxruntime::GraphViewer& graph,
          const std::vector<const KernelRegistry*>& kernel_registries) const override;

  const void* GetExecutionHandle() const noexcept override {
    // The ARM interface does not return anything interesting.
    return nullptr;
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
};

}  // namespace onnxruntime
