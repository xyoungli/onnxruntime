// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"
#include "core/platform/threadpool.h"
#include "core/providers/arm/arm_execution_provider.h"

namespace onnxruntime {
namespace arm {

using ActivationFuncPtr = void (*)(const float* din, float* dout, int size);
ActivationFuncPtr ActivationFuncByName(const std::string& act_name);

/// The class represents ARM implementation of a long short term memory (LSTM) operator.
/// For details, refer to http://aka.ms/dl-optimization/.
class LstmOp final : public OpKernel {
public:
  LstmOp(const OpKernelInfo &info)
          : OpKernel(info) {
    provider_ = const_cast<ARMExecutionProvider*>(
            dynamic_cast<const ARMExecutionProvider*>(info.GetExecutionProvider()));
    ORT_ENFORCE(provider_ != nullptr);
    alloc_ptr_ = provider_->GetAllocator(0, OrtMemTypeDefault);
    out_iofc_ = std::make_shared<Buffer>(alloc_ptr_);
    hidden_state_ = std::make_shared<Buffer>(alloc_ptr_);
    cell_state_ = std::make_shared<Buffer>(alloc_ptr_);
    i_ = std::make_shared<Buffer>(alloc_ptr_);
    o_ = std::make_shared<Buffer>(alloc_ptr_);
    f_ = std::make_shared<Buffer>(alloc_ptr_);
    c_ = std::make_shared<Buffer>(alloc_ptr_);
    clip_ = info.GetAttrOrDefault<float>("clip", std::numeric_limits<float>::max());
    ORT_ENFORCE(clip_ > 0.f);

    std::string direction;
    ORT_ENFORCE(info.GetAttr("direction", &direction).IsOK());

    int64_t int64_value;
    ORT_ENFORCE(info.GetAttr("hidden_size", &int64_value).IsOK() && int64_value > 0);
    hidden_size_ = static_cast<int>(int64_value);

    // optional attributes
    std::vector<std::string> activation_func_names = info.GetAttrsOrDefault<std::string>("activations");
    std::vector<float> activation_func_alphas = info.GetAttrsOrDefault<float>("activation_alpha");
    std::vector<float> activation_func_betas = info.GetAttrsOrDefault<float>("activation_beta");

    if (info.GetAttr("input_forget", &int64_value).IsOK())
      input_forget_ = int64_value != 0;

    direction_ = rnn::detail::MakeDirection(direction);
    num_directions_ = direction_ == rnn::detail::Direction::kBidirectional ? 2 : 1;

    if (activation_func_names.empty()) {
      for (int i = 0; i < num_directions_; ++i) {
        activation_func_names.emplace_back("sigmoid");
        activation_func_names.emplace_back("tanh");
        activation_func_names.emplace_back("tanh");
      }
    }

    ORT_ENFORCE(activation_func_names.size() == static_cast<size_t>(num_directions_) * 3);

    for (auto& act_name : activation_func_names) {
      act_funcs_.push_back(ActivationFuncByName(act_name));
    }
  }

  Status Compute(OpKernelContext *context) const override;

  ~LstmOp() override {}

private:
  Status ComputeImplFP32(OpKernelContext &context) const;
  Status PrepareWorkspace(OpKernelContext &context) const;
  Status PrepareWeights(OpKernelContext &context) const;
  Status ValidateInputs(const Tensor &X,
                        const Tensor &W,
                        const Tensor &R,
                        const Tensor *B,
                        const Tensor *sequence_lens,
                        const Tensor *initial_h,
                        const Tensor *initial_c,
                        const Tensor *P,
                        int batch_size) const;

  Status GateCompute(const float* seq_iofc, const float* pre_cell_ptr, float* cell_ptr, float* hiddhen_ptr) const;

  mutable std::shared_ptr<Buffer> out_iofc_{nullptr};
  mutable std::shared_ptr<Buffer> hidden_state_{nullptr};
  mutable std::shared_ptr<Buffer> cell_state_{nullptr};
  mutable std::shared_ptr<Buffer> i_{nullptr};
  mutable std::shared_ptr<Buffer> o_{nullptr};
  mutable std::shared_ptr<Buffer> f_{nullptr};
  mutable std::shared_ptr<Buffer> c_{nullptr};

  mutable std::vector<int> vseq_len_{};
  mutable int seq_len_sum_{0};
  mutable int seq_len_{0};
  mutable int max_seq_len_{0};
  mutable int batch_size_{0};
  mutable int input_size_{0};

  ARMExecutionProvider* provider_{nullptr};
  AllocatorPtr alloc_ptr_{nullptr};
  rnn::detail::Direction direction_;
  int num_directions_;

  int hidden_size_ = 0;
  float clip_;
  bool input_forget_ = false;

  std::vector<ActivationFuncPtr> act_funcs_{};
};

}  // namespace arm
}  // namespace onnxruntime
