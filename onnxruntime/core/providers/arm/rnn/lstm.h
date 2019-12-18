// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace arm {
/// The class represents ARM implementation of a long short term memory (LSTM) operator.
/// For details, refer to http://aka.ms/dl-optimization/.
class LstmOp final : public OpKernel {
public:
  LstmOp(const OpKernelInfo &info)
          : OpKernel(info), clip_(info.GetAttrOrDefault<float>("clip", std::numeric_limits<float>::max())) {
    std::string direction;
    ORT_ENFORCE(info.GetAttr("direction", &direction).IsOK());

    int64_t int64_value;
    ORT_ENFORCE(info.GetAttr("hidden_size", &int64_value).IsOK() && int64_value > 0);
    hidden_size_ = gsl::narrow<int>(int64_value);

    // optional attributes
    std::vector<std::string> activation_func_names = info.GetAttrsOrDefault<std::string>("activations");
    std::vector<float> activation_func_alphas = info.GetAttrsOrDefault<float>("activation_alpha");
    std::vector<float> activation_func_betas = info.GetAttrsOrDefault<float>("activation_beta");
    ORT_ENFORCE(clip_ > 0.f);

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

    activation_funcs_ = rnn::detail::ActivationFuncs(activation_func_names,
                                                     activation_func_alphas,
                                                     activation_func_betas);
  }

  Status Compute(OpKernelContext *context) const override;

  ~LstmOp() override = default;

private:
  template<typename T>
  Status ComputeImpl(OpKernelContext &context) const;

  Status ValidateInputs(const Tensor &X,
                        const Tensor &W,
                        const Tensor &R,
                        const Tensor *B,
                        const Tensor *sequence_lens,
                        const Tensor *initial_h,
                        const Tensor *initial_c,
                        const Tensor *P,
                        int batch_size) const;

  rnn::detail::Direction direction_;
  int num_directions_;

  int hidden_size_ = 0;
  float clip_;
  bool input_forget_ = false;

  rnn::detail::ActivationFuncs activation_funcs_;

  // Threadpool for operator. If concurrent Compute calls are possible, it will be shared
  // across them. mutable due to this.
  // The alternative would be to create a threadpool in each call to Compute but that would incur thread creation
  // cost on every call.
  mutable onnxruntime::concurrency::ThreadPool lstm_tp_{"LSTM",
                                                        static_cast<int>(std::thread::hardware_concurrency())};
};

}  // namespace arm
}  // namespace onnxruntime
