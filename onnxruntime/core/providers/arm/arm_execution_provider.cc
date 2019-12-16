// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/arm/arm_execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"

#include "core/framework/compute_capability.h"

constexpr const char* ARM = "Arm";

namespace onnxruntime {
namespace arm {
// Forward declarations of op kernels
//class ONNX_OPERATOR_KERNEL_CLASS_NAME(kArmExecutionProvider, kOnnxDomain, 7, LSTM);

Status RegisterArmKernels(KernelRegistry &kernel_registry) {
//  static const BuildKernelCreateInfoFn function_table[] = {
//          BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kArmExecutionProvider, kOnnxDomain, 7, LSTM)>,
//  };
//
//  for (auto &function_table_entry : function_table) {
//    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
//  }
  return Status::OK();
}

std::shared_ptr<KernelRegistry> GetArmKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterArmKernels(*kernel_registry);
  return kernel_registry;
}

}  // namespace arm

ARMExecutionProvider::ARMExecutionProvider(const ARMExecutionProviderInfo& info)
        : IExecutionProvider{onnxruntime::kAclExecutionProvider} {
  ORT_UNUSED_PARAMETER(info);

  auto default_allocator_factory = [](int) {
    auto memory_info = onnxruntime::make_unique<OrtMemoryInfo>(CPU, OrtAllocatorType::OrtDeviceAllocator);
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
}

ARMExecutionProvider::~ARMExecutionProvider() {}

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

}  // namespace onnxruntime
