// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <DirectML.h>
#include <dxgi1_4.h>

#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

#include <wil/wrl.h>
#include <wil/result.h>

#include "core/providers/dml/dml_provider_factory.h"
#include "core/session/abi_session_options_impl.h"
#include "DmlExecutionProvider/inc/DmlExecutionProvider.h"

namespace onnxruntime {

struct DMLProviderFactory : IExecutionProviderFactory {
  DMLProviderFactory(IDMLDevice* dml_device,
                     ID3D12CommandQueue* cmd_queue) : dml_device_(dml_device),
                                                      cmd_queue_(cmd_queue) {}
  ~DMLProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  ComPtr<IDMLDevice> dml_device_{};
  ComPtr<ID3D12CommandQueue> cmd_queue_{};
};

std::unique_ptr<IExecutionProvider> DMLProviderFactory::CreateProvider() {
  return Dml::CreateExecutionProvider(dml_device_.Get(), cmd_queue_.Get());
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_DML(IDMLDevice* dml_device,
                                                                              ID3D12CommandQueue* cmd_queue) {
  // Validate that the D3D12 devices match between DML and the command queue. This specifically asks for IUnknown in
  // order to be able to compare the pointers for COM object identity.
  ComPtr<IUnknown> d3d12_device_0;
  ComPtr<IUnknown> d3d12_device_1;
  THROW_IF_FAILED(dml_device->GetParentDevice(IID_PPV_ARGS(&d3d12_device_0)));
  THROW_IF_FAILED(cmd_queue->GetDevice(IID_PPV_ARGS(&d3d12_device_1)));

  if (d3d12_device_0 != d3d12_device_1) {
    THROW_HR(E_INVALIDARG);
  }

  return std::make_shared<onnxruntime::DMLProviderFactory>(dml_device, cmd_queue);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_DML(int device_id) {
  ComPtr<IDXGIFactory4> dxgi_factory;
  THROW_IF_FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgi_factory)));

  ComPtr<IDXGIAdapter1> adapter;
  THROW_IF_FAILED(dxgi_factory->EnumAdapters1(device_id, &adapter));

  ComPtr<ID3D12Device> d3d12_device;
  THROW_IF_FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12_device)));

  D3D12_COMMAND_QUEUE_DESC cmd_queue_desc = {};
  cmd_queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  cmd_queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  
  ComPtr<ID3D12CommandQueue> cmd_queue;
  THROW_IF_FAILED(d3d12_device->CreateCommandQueue(&cmd_queue_desc, IID_PPV_ARGS(&cmd_queue)));

  DML_CREATE_DEVICE_FLAGS flags = DML_CREATE_DEVICE_FLAG_NONE;

  // In debug builds, enable the DML debug layer if the D3D12 debug layer is also enabled
#if _DEBUG
  ComPtr<ID3D12DebugDevice> debug_device;
  (void)d3d12_device->QueryInterface(IID_PPV_ARGS(&debug_device)); // ignore failure
  const bool is_d3d12_debug_layer_enabled = (debug_device != nullptr);

  if (is_d3d12_debug_layer_enabled) {
    flags |= DML_CREATE_DEVICE_FLAG_DEBUG;
  }
#endif

  ComPtr<IDMLDevice> dml_device;
  THROW_IF_FAILED(DMLCreateDevice1(d3d12_device.Get(),
                                   flags,
                                   DML_FEATURE_LEVEL_2_0,
                                   IID_PPV_ARGS(&dml_device)));

  return CreateExecutionProviderFactory_DML(dml_device.Get(), cmd_queue.Get());
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_DML, _In_ OrtSessionOptions* options, int device_id) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_DML(device_id));
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProviderEx_DML, _In_ OrtSessionOptions* options,
                    IDMLDevice* dml_device, ID3D12CommandQueue* cmd_queue) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_DML(dml_device,
                                                                                        cmd_queue));
  return nullptr;
}
