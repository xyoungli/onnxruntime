// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/arm/arm_execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"

#include "core/framework/compute_capability.h"

namespace onnxruntime {
namespace arm {
// Forward declarations of op kernels
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kArmExecutionProvider, kOnnxDomain, 7, LSTM);

Status RegisterArmKernels(KernelRegistry &kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
          BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kArmExecutionProvider, kOnnxDomain, 7, LSTM)>,
  };

  for (auto &function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }
  return Status::OK();
}

std::shared_ptr<KernelRegistry> GetArmKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterArmKernels(*kernel_registry);
  return kernel_registry;
}

}  // namespace arm

ARMExecutionProvider::ARMExecutionProvider(const ARMExecutionProviderInfo& info)
        : IExecutionProvider{onnxruntime::kArmExecutionProvider} {
  mode_ = info.mode;
  threads_ = info.threads;
  arm::ARMDevice::Global().Init();

  auto default_allocator_factory = [](int) {
    auto memory_info = onnxruntime::make_unique<OrtMemoryInfo>(ARM, OrtAllocatorType::OrtDeviceAllocator);
    return onnxruntime::make_unique<CPUAllocator>(std::move(memory_info));
  };

  DeviceAllocatorRegistrationInfo default_memory_info{
          OrtMemTypeDefault,
          std::move(default_allocator_factory),
          std::numeric_limits<size_t>::max()};

  InsertAllocator(CreateAllocator(default_memory_info));

  auto cpu_allocator_factory = [](int) {
    auto memory_info = onnxruntime::make_unique<OrtMemoryInfo>(
            ARM, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput);
    return onnxruntime::make_unique<CPUAllocator>(std::move(memory_info));
  };

  DeviceAllocatorRegistrationInfo cpu_memory_info{
          OrtMemTypeCPUOutput,
          std::move(cpu_allocator_factory),
          std::numeric_limits<size_t>::max()};

  InsertAllocator(CreateAllocator(cpu_memory_info));
  workspace_ = std::make_shared<Buffer>(GetAllocator(0, OrtMemTypeDefault));
  SetRunMode(mode_, threads_);
}

std::unique_ptr<IDataTransfer> ARMExecutionProvider::GetDataTransfer() const {
  return onnxruntime::make_unique<CPUDataTransfer>();
}

ARMExecutionProvider::~ARMExecutionProvider() {
  ClearWorkspace();
}

std::shared_ptr<KernelRegistry> ARMExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::arm::GetArmKernelRegistry();
  return kernel_registry;
}

std::vector<std::unique_ptr<ComputeCapability>>
ARMExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                    const std::vector<const KernelRegistry*>& kernel_registries) const {
  std::vector<std::unique_ptr<ComputeCapability>>
          result = IExecutionProvider::GetCapability(graph, kernel_registries);

  return result;
}

void ARMExecutionProvider::RequestPowerFullMode(int threads) {
  auto& dev = arm::ARMDevice::Global().Info();
  int big_core_size = dev.big_core_ids.size();
  int little_core_size = dev.little_core_ids.size();
  active_ids_.clear();
  for (int i = 0; i < threads; ++i) {
    if (i < big_core_size) {
      active_ids_.push_back(dev.big_core_ids[i]);
    } else if (i < big_core_size + little_core_size) {
      active_ids_.push_back(dev.little_core_ids[i - big_core_size]);
    }
  }
  mode_ = PowerMode::ARM_POWER_FULL;
}

void ARMExecutionProvider::RequestPowerHighMode(int threads) {
  auto& dev = arm::ARMDevice::Global().Info();
  int big_core_size = dev.big_core_ids.size();
  int little_core_size = dev.little_core_ids.size();
  active_ids_.clear();
  if (big_core_size > 0) {
    mode_ = PowerMode::ARM_POWER_HIGH;
    if (threads > big_core_size) {
      printf("Request thread num: %d, exceed the big cores size: %d, truncate thread num to %d\n",
              threads, big_core_size, big_core_size);
      active_ids_ = dev.big_core_ids;
    } else {
      for (int i = 0; i < threads; ++i) {
        active_ids_.push_back(dev.big_core_ids[big_core_size - 1 - i]);
      }
    }
  } else {
    mode_ = PowerMode::ARM_POWER_LOW;
    printf("HIGH POWER MODE is not support, switch to little cores.\n");
    if (threads > little_core_size) {
      active_ids_ = dev.little_core_ids;
    } else {
      for (int i = 0; i < threads; ++i) {
        active_ids_.push_back(dev.little_core_ids[i]);
      }
    }
  }
}

void ARMExecutionProvider::RequestPowerLowMode(int threads) {
  auto& dev = arm::ARMDevice::Global().Info();
  int big_core_size = dev.big_core_ids.size();
  int little_core_size = dev.little_core_ids.size();
  active_ids_.clear();
  if (little_core_size > 0) {
    mode_ = PowerMode::ARM_POWER_LOW;
    if (threads > little_core_size) {
      printf("Request thread num: %d, exceed the little cores size: %d, truncate thread num to %d\n",
      threads, little_core_size, little_core_size);
      active_ids_ = dev.little_core_ids;
    } else {
      for (int i = 0; i < threads; i++) {
        active_ids_.push_back(dev.little_core_ids[i]);
      }
    }
  } else {
    mode_ = PowerMode::ARM_POWER_HIGH;
    printf("LOW POWER MODE is not support, switch to big cores.\n");
    if (threads > big_core_size) {
      active_ids_ = dev.big_core_ids;
    } else {
      for (int i = 0; i < threads; i++) {
        active_ids_.push_back(dev.big_core_ids[i]);
      }
    }
  }
}

void ARMExecutionProvider::RequestPowerNoBindMode(int threads) {
  auto& dev = arm::ARMDevice::Global().Info();
  active_ids_.clear();
  if (threads > dev.core_num) {
    active_ids_ = dev.cluster_ids;
  } else {
    active_ids_.resize(threads);
    for (int i = 0; i < threads; ++i) {
      if (i < dev.big_core_ids.size()) {
        active_ids_[i] = dev.big_core_ids[i];
      } else {
        active_ids_[i] = dev.little_core_ids[i - dev.big_core_ids.size()];
      }
    }
  }
  mode_ = PowerMode::ARM_POWER_NO_BIND;
}

void ARMExecutionProvider::SetRunMode(PowerMode mode, int threads) {
#ifndef USE_OPENMP
  threads_ = 1;  // force thread_num to 1 if OpenMP is disabled
  active_ids_ = {0};
#else
  switch (mode) {
    case ARM_POWER_FULL:
      RequestPowerFullMode(threads);
      break;
    case ARM_POWER_HIGH:
      RequestPowerHighMode(threads);
      break;
    case ARM_POWER_LOW:
      RequestPowerLowMode(threads);
      break;
    case ARM_POWER_NO_BIND:
      RequestPowerNoBindMode(threads);
      break;
    default:
      printf("Unsupported power mode: %d\n", static_cast<int>(mode));
      break;
  }
  if (active_ids_.empty()) {
    active_ids_.push_back(0);
  }
  threads_ = active_ids_.size();
#if defined(USE_ARM_LINUX) || defined(USE_ANDROID)
  omp_set_num_threads(active_ids_.size());
  if (mode_ == ARM_POWER_LOW || (mode_ == ARM_POWER_HIGH && threads_ == 1)) {
    if (arm::CheckCpuOnline(active_ids_)) {
      arm::BindThreads(active_ids_);
    } else {
      printf("Some cores are offline, switch to NO BIND MODE\n");
      mode_ = ARM_POWER_NO_BIND;
      omp_set_num_threads(threads_);
    }
  } else {
    omp_set_num_threads(threads_);
  }
#else
  RequestPowerNoBindMode(threads_);
  omp_set_num_threads(threads_);
#endif  //  USE_ARM_LINUX OR USE_ANDROID
#endif  //  USE_OPENMP
  auto& dev = arm::ARMDevice::Global().Info();
  int id = active_ids_[0];
  arch_ = dev.archs[id];
  has_dot_ = dev.dot[id];
  has_fp16_ = dev.fp16[id];
  l1_size_ = dev.L1_cache[id];
  l2_size_ = dev.L2_cache[id];
  l3_size_ = dev.L3_cache[id];
  llc_size_ = arm::ARMDevice::Global().LLCSize(id);
  ExtendWorkspace(llc_size_);
}

void ARMExecutionProvider::ClearWorkspace() {
  workspace_->Clean();
}

void ARMExecutionProvider::SetArch(int arch) {
  arch_ = static_cast<arm::ARMArch>(arch);
}

void ARMExecutionProvider::SetCache(int L1_size, int L2_size, int L3_size) {
  l1_size_ = L1_size;
  l2_size_ = L2_size;
  l3_size_ = L3_size;
  llc_size_ = l3_size_ > 0? l3_size_ : l2_size_;
  ExtendWorkspace(llc_size_);
}

bool ARMExecutionProvider::ExtendWorkspace(size_t size) {
  return workspace_->ReAlloc(size);
}

arm::ARMArch ARMExecutionProvider::Arch() const {
  return arch_;
}

PowerMode ARMExecutionProvider::Mode() const {
  return mode_;
}

int ARMExecutionProvider::Threads() const {
  return threads_;
}

bool ARMExecutionProvider::HasDot() const {
  return has_dot_;
}

bool ARMExecutionProvider::HasFp16() const {
  return has_fp16_;
}

int ARMExecutionProvider::L1CacheSize() const {
  return l1_size_;
}

int ARMExecutionProvider::L2CacheSize() const {
  return l2_size_;
}

int ARMExecutionProvider::L3CacheSize() const {
  return l3_size_;
}

int ARMExecutionProvider::LLCSize() const {
  return llc_size_;
}

Buffer::Buffer(AllocatorPtr ptr, size_t size) {
  alloc_ = ptr;
  capacity_ = size;
  count_ = size;
  owndata_ = true;
  data_ = alloc_->Alloc(size);
}

Buffer::Buffer(AllocatorPtr ptr, void *data, size_t size) {
  alloc_ = ptr;
  capacity_ = size;
  count_ = size;
  data_ = data;
  owndata_ = false;
}


bool Buffer::MemSet(int c, size_t size) {
  if (!owndata_ || count_ < size) {
    return false;
  }
  std::memset(data_, c, size);
  return true;
}

bool Buffer::ReAlloc(size_t size) {
  if (size > capacity_) {
    if (owndata_) {
      Clean();
      data_ = alloc_->Alloc(size);
      capacity_ = size;
    } else {
      return false;
    }
  }
  count_ = size;
  return true;
}

bool Buffer::CopyFrom(const Buffer &buf, size_t size) {
  std::memcpy(data_, buf.data_, size);
  return true;
}

bool Buffer::Clean() {
  if (owndata_ && capacity_ > 0) {
    count_ = 0;
    capacity_ = 0;
    owndata_ = true;
    alloc_->Free(data_);
  }
  data_ = nullptr;
  return true;
}

}  // namespace onnxruntime
