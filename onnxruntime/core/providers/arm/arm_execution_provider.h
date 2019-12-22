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

class Buffer {
public:
  explicit Buffer(AllocatorPtr ptr) {
    ORT_ENFORCE(ptr);
    alloc_ = ptr;
  }
  Buffer(const Buffer &buf) = delete;
  Buffer(const Buffer &&buf) = delete;
  Buffer &operator=(const Buffer &buf) = delete;
  Buffer &operator=(const Buffer &&buf) = delete;

  /**
   * \brief constructor with buffer size, in Dtype
   */
  explicit Buffer(AllocatorPtr ptr, size_t size);
  explicit Buffer(AllocatorPtr ptr, void *data, size_t size);

  /**
   * \brief destructor
   */
  ~Buffer() { Clean(); }

  /**
   * \brief set each bytes of data_ to (c) with length of (size)
   */
  bool MemSet(int c, size_t size);

  /**
   * \brief re-alloc memory, only if hold the data, can be relloc
   */
  bool ReAlloc(size_t size);

  /**
   * \brief sync copy from other buffer
   * @param buf
   * @return
   */
  bool CopyFrom(const Buffer &buf, size_t size);

  /**
   * \brief return const data pointer
   */
  template <typename T>
  const T *Data() const {
    return reinterpret_cast<const T*>(data_);
  }

  /**
   * \brief return mutable data pointer
   */
  template <typename T>
  T *MutableData() {
    return reinterpret_cast<T*>(data_);
  }

  /**
   * \brief free memory
   */
  bool Clean();

  /**
   * \brief return total size of memory, in size
   */
  inline size_t Capacity() const { return capacity_; }

private:
  AllocatorPtr alloc_;
  void *data_{nullptr};
  bool owndata_{true};
  size_t count_{0};
  size_t capacity_{0};
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
  T* WorkspaceData() {
    return reinterpret_cast<T*>(workspace_->MutableData<T>());
  }

private:
  std::shared_ptr<Buffer> workspace_{nullptr};
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
