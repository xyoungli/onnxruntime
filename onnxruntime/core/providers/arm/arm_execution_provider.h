// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"
#include "arm_device_info.h"

namespace onnxruntime {
// Information needed to construct ARM execution providers.
struct ARMExecutionProviderInfo {
  bool create_arena{true};
  PowerMode mode{PowerMode::ARM_POWER_NO_BIND};
  int threads{1};

  explicit ARMExecutionProviderInfo(bool use_arena, PowerMode power_mode, int num_threads=1)
          : create_arena(use_arena), mode(power_mode), threads(num_threads) {}

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
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

  void SetRunMode(PowerMode mode, int threads);
  void SetCache(int L1_size, int L2_size, int L3_size);
  void SetArch(int arch);
  bool ExtendWorkspace(size_t size);
  void ClearWorkspace();
  int L1CacheSize() const;
  int L2CacheSize() const;
  int L3CacheSize() const;
  int LLCSize() const;
  bool HasDot() const;
  bool HasFp16() const;
  arm::ARMArch Arch() const;
  int Threads() const;
  PowerMode Mode() const;

  template <typename T>
  T* workspace_data() {
    return reinterpret_cast<T*>(workspace_data_);
  }

private:
  void* workspace_data_{nullptr};
  size_t workspace_size_{0};
  arm::ARMArch arch_;
  PowerMode mode_{ARM_POWER_NO_BIND};
  uint32_t threads_{1};
  bool has_dot_{false};
  bool has_fp16_{false};
  int l1_size_{32 * 1024};
  int l2_size_{512 * 1024};
  int l3_size_{0};
  int llc_size_{512 * 1024};
  std::vector<int> active_ids_{0};
  void RequestPowerNoBindMode(int threads);
  void RequestPowerHighMode(int threads);
  void RequestPowerLowMode(int threads);
  void RequestPowerFullMode(int threads);
};

}  // namespace onnxruntime
