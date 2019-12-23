// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "core/platform/threadpool.h"
#include "core/framework/op_kernel_context_internal.h"

#include "core/providers/arm/rnn/lstm.h"

#include "core/common/common.h"

#include "core/providers/arm/funcs/sgemm.h"
#include "core/providers/arm/funcs/activation.h"
#include "core/providers/arm/funcs/split.h"
#include "core/providers/arm/funcs/elementwise.h"
#include "test/data_utils.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

/*
ONNX_OPERATOR_SCHEMA(LSTM)
    .SetDoc(R"DOC(
Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

`X` - input tensor
`i` - input gate
`o` - output gate
`f` - forget gate
`c` - cell gate
`t` - time step (t-1 means previous time step)

`W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates
`R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates
`Wb[iofc]` - W bias vectors for input, output, forget, and cell gates
`Rb[iofc]` - R bias vectors for input, output, forget, and cell gates
`P[iof]`  - P peephole weight vector for input, output, and forget gates
`WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates
`RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates
`WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates
`RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates
`PB[iof]`  - P peephole weight vector for backward input, output, and forget gates

`H` - Hidden state
`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)
  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)
  Affine(x)              - alpha*x + beta
  LeakyRelu(x)           - x if x >= 0 else alpha * x
  ThresholdedRelu(x)     - x if x >= alpha else 0
  ScaledTanh(x)          - alpha*Tanh(beta*x)
  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
  Softsign(x)            - x/(1 + |x|)
  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):
  - it = f(Xt*(Wi^T) + Ht-1*Ri + Pi (.) Ct-1 + Wbi + Rbi)
  - ft = f(Xt*(Wf^T) + Ht-1*Rf + Pf (.) Ct-1 + Wbf + Rbf)
  - ct = g(Xt*(Wc^T) + Ht-1*Rc + Wbc + Rbc)
  - Ct = ft (.) Ct-1 + it (.) ct
  - ot = f(Xt*(Wo^T) + Ht-1*Ro + Po (.) Ct + Wbo + Rbo)
  - Ht = ot (.) h(Ct)
)DOC")
    .Attr("direction", "Specify if the RNN is forward, reverse, or bidirectional. "
               "Must be one of forward (default), reverse, or bidirectional.",
               AttributeProto::STRING,
               std::string("forward"))
    .Attr("hidden_size", "Number of neurons in the hidden layer", AttributeProto::INT, OPTIONAL)
    .Attr("activations", "A list of 3 (or 6 if bidirectional) activation functions "
               "for input, output, forget, cell, and hidden. The activation functions must "
               "be one of the activation functions specified above. Optional: See the equations "
               "for default if not specified.",
               AttributeProto::STRINGS,
               OPTIONAL)
    .Attr("activation_alpha",
               "Optional scaling values used by some activation functions. The values "
               "are consumed in the order of activation functions, for example (f, g, h) "
               "in LSTM.",
               AttributeProto::FLOATS,
               OPTIONAL)
    .Attr("activation_beta",
               "Optional scaling values used by some activation functions. The values "
               "are consumed in the order of activation functions, for example (f, g, h) "
               "in LSTM.",
               AttributeProto::FLOATS,
               OPTIONAL)
    .Attr("output_sequence",
               "The sequence output for the hidden is optional if 0. Default 0.",
               AttributeProto::INT,
               static_cast<int64_t>(0));
    .Attr("clip", "Cell clip threshold. Clipping bounds the elements of a tensor "
               "in the range of [-threshold, +threshold] and is applied to the input "
               "of activations. No clip if not specified.", AttributeProto::FLOAT, OPTIONAL)
    .Attr("input_forget", "Couple the input and forget gates if 1, default 0.",
               AttributeProto::INT,
               static_cast<int64_t>(0))
    .Input(0, "X",
               "The input sequences packed (and potentially padded) into one 3-D "
               "tensor with the shape of `[seq_length, batch_size, input_size]`.", "T")
    .Input(1, "W",
               "The weight tensor for the gates. Concatenation of `W[iofc]` and "
               "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
               "`[num_directions, 4*hidden_size, input_size]`.", "T")
    .Input(2, "R",
               "The recurrence weight tensor. Concatenation of `R[iofc]` and "
               "`RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape "
               "`[num_directions, 4*hidden_size, hidden_size]`.", "T")
    .Input(3, "B",
               "The bias tensor for input gate. Concatenation of `[Wb[iofc], Rb[iofc]]`, "
               "and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0. This "
               "tensor has shape `[num_directions, 8*hidden_size]`. Optional: If not "
               "specified - assumed to be 0.", "T",
               OpSchema::Optional)
    .Input(4, "sequence_lens",
               "Optional tensor specifying lengths of the sequences in a batch. "
               "If not specified - assumed all sequences in the batch to have "
               "length `seq_length`. It has shape `[batch_size]`.", "T1",
               OpSchema::Optional)
    .Input(5, "initial_h",
                "Optional initial value of the hidden. If not specified - assumed "
                "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
                "T", OpSchema::Optional)
    .Input(6, "initial_c",
                "Optional initial value of the cell. If not specified - assumed "
                "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
"T", OpSchema::Optional)
    .Input(7, "P",
                "The weight tensor for peepholes. Concatenation of `P[iof]` and "
                "`PB[iof]` (if bidirectional) along dimension 0. It has shape "
                "`[num_directions, 3*hidde_size]`. Optional: If not specified - "
                "assumed to be 0.", "T",
                OpSchema::Optional)
    .Output(0, "Y",
                "A tensor that concats all the intermediate output values of the hidden. "
                "It has shape `[seq_length, num_directions, batch_size, hidden_size]`. ",
                "T", OpSchema::Optional);
    .Output(1, "Y_h",
                "The last output value of the hidden. It has shape "
                "`[num_directions, batch_size, hidden_size]`.", "T", OpSchema::Optional);
    .Output(2, "Y_c",
                "The last output value of the cell. It has shape "
                "`[num_directions, batch_size, hidden_size]`.", "T", OpSchema::Optional);
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                    "Constrain input and output types to float tensors.")
    .TypeConstraint("T1", { "tensor(int32)" }, "Constrain seq_lens to integer tensor.");

*/

namespace onnxruntime {
namespace arm {

/* LSTM operator */
ONNX_OPERATOR_KERNEL_EX(
      LSTM,
      kOnnxDomain,
      7,
      kArmExecutionProvider,
      KernelDefBuilder()
          .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
      LstmOp);

ActivationFuncPtr ActivationFuncByName(const std::string& act_name) {
  if (act_name == "sigmoid") {
    return funcs::ActSigmoid;
  }
  if (act_name == "tanh") {
    return funcs::ActTanh;
  }
  ORT_THROW("Invalid activation type for LSTM operator of ", act_name);
  return nullptr;
}

void ReverseSequence(const float* inputs,
                     float* inputs_reverse,
                     const std::vector<int>& sequence_lengths,
                     int max_sequence_length,
                     int batch_size,
                     int input_size,
                     int num_directions) {
  for (int i = 0; i < batch_size; i++) {
    int seq_len = sequence_lengths[i];
#pragma omp parallel for
    for (int j = 0; j < seq_len; j++) {
      auto src = inputs + j * batch_size * input_size + i * input_size;
      auto dest = inputs_reverse + num_directions * (seq_len - j - 1) * batch_size * input_size + i * input_size;
      std::memcpy(dest, src, input_size * sizeof(float));
    }
#pragma omp parallel for
    for (int j = seq_len; j < max_sequence_length; j++) {
      auto  src = inputs + j * batch_size * input_size + i * input_size;
      auto dest = inputs_reverse + num_directions * j * batch_size * input_size + i * input_size;
      std::memcpy(dest, src, input_size * sizeof(float));
    }
  }
}

void GateClip(const float* din, float* dout, float clip, int size) {
#ifdef USE_OPENMP
  int threads = omp_get_num_threads();
#else
  int threads = 1;
#endif
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt = nums_per_thread >> 4;
  int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
  float32x4_t vclip = vdupq_n_f32(clip);
  float clip_minus = -1.f * clip;
  float32x4_t vclip_minus = vdupq_n_f32(clip_minus);
#pragma omp parallel for
  for (int i = 0; i < threads; ++i) {
    int idx_start = i * nums_per_thread;
    const float* ptr_in_thread = din + idx_start;
    float* ptr_out_thread = dout + idx_start;
    for (int num = 0; num < neon_loop_cnt; ++num) {

      vst1q_f32(ptr_out_thread, vmaxq_f32(vminq_f32(vld1q_f32(ptr_in_thread), vclip), vclip_minus));
      vst1q_f32(ptr_out_thread + 4, vmaxq_f32(vminq_f32(vld1q_f32(ptr_in_thread + 4), vclip), vclip_minus));
      vst1q_f32(ptr_out_thread + 8, vmaxq_f32(vminq_f32(vld1q_f32(ptr_in_thread + 8), vclip), vclip_minus));
      vst1q_f32(ptr_out_thread + 12, vmaxq_f32(vminq_f32(vld1q_f32(ptr_in_thread + 12), vclip), vclip_minus));
      ptr_in_thread += 16;
      ptr_out_thread += 16;
    }
    for (int j = 0; j < neon_loop_remain; ++j) {
      *ptr_out_thread = std::max(std::min(*ptr_in_thread, clip), clip_minus);
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  float* out_ptr_remain = dout + threads * nums_per_thread;
  const float* in_ptr_remain = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    *out_ptr_remain = std::max(std::min(*in_ptr_remain, clip), clip_minus);
    in_ptr_remain++;
    out_ptr_remain++;
  }
}

void InputForget(const float* din, float* dout, int size) {
#ifdef USE_OPENMP
  int threads = omp_get_num_threads();
#else
  int threads = 1;
#endif
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt = nums_per_thread >> 4;
  int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
  float32x4_t vones = vdupq_n_f32(1.f);
#pragma omp parallel for
  for (int i = 0; i < threads; ++i) {
    int idx_start = i * nums_per_thread;
    const float* ptr_in_thread = din + idx_start;
    float* ptr_out_thread = dout + idx_start;
    for (int num = 0; num < neon_loop_cnt; ++num) {
      vst1q_f32(ptr_out_thread, vsubq_f32(vones, vld1q_f32(ptr_in_thread)));
      vst1q_f32(ptr_out_thread + 4, vsubq_f32(vones, vld1q_f32(ptr_in_thread + 4)));
      vst1q_f32(ptr_out_thread + 8, vsubq_f32(vones, vld1q_f32(ptr_in_thread + 8)));
      vst1q_f32(ptr_out_thread + 12, vsubq_f32(vones, vld1q_f32(ptr_in_thread + 12)));
      ptr_in_thread += 16;
      ptr_out_thread += 16;
    }
    for (int j = 0; j < neon_loop_remain; ++j) {
      *ptr_out_thread = 1.f - *ptr_in_thread;
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  float* out_ptr_remain = dout + threads * nums_per_thread;
  const float* in_ptr_remain = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    *out_ptr_remain = 1.f - *in_ptr_remain;
    in_ptr_remain++;
    out_ptr_remain++;
  }
}

Status LstmOp::Compute(OpKernelContext *context) const {
  if (!weights_init_) {
    PrepareWeights(*context);
    weights_init_ = true;
  }
  PrepareWorkspace(*context);
  const Tensor &X = *context->Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]
  Status status;
  if (X.IsDataType<float>()) {
    status = ComputeImplFP32(*context);
  } else if (X.IsDataType<double>()) {
    /* Need to update all the helpers to support double...
    status = ComputeImpl<double>(*context); */
    ORT_NOT_IMPLEMENTED("LSTM operator does not support double yet");
  } else {
    ORT_THROW("Invalid data type for LSTM operator of ", X.DataType());
  }
  return status;
}

Status LstmOp::ValidateInputs(const Tensor &X, const Tensor &W, const Tensor &R, const Tensor *B,
                              const Tensor *sequence_lens, const Tensor *initial_h, const Tensor *initial_c,
                              const Tensor *P, int batch_size) const {
  auto status = rnn::detail::ValidateCommonRnnInputs(X, W, R, B, 4, sequence_lens, initial_h,
                                                     num_directions_, hidden_size_);
  ORT_RETURN_IF_ERROR(status);

  if (initial_c != nullptr) {
    auto &initial_c_shape = initial_c->Shape();

    if (initial_c_shape.NumDimensions() != 3 ||
        initial_c_shape[0] != num_directions_ ||
        initial_c_shape[1] != batch_size ||
        initial_c_shape[2] != hidden_size_)

      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input initial_c must have shape {",
                             num_directions_, ",", batch_size, ",", hidden_size_, "}. Actual:", initial_c_shape);
  }

  if (P != nullptr) {
    auto &p_shape = P->Shape();

    if (p_shape.NumDimensions() != 2 ||
        p_shape[0] != num_directions_ ||
        p_shape[1] != 3 * hidden_size_)

      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input P must have shape {",
                             num_directions_, ",", 3 * hidden_size_, "}. Actual:", p_shape);
  }

  return Status::OK();
}

Status LstmOp::PrepareWorkspace(onnxruntime::OpKernelContext &context) const {
  const Tensor &X = *context.Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]
  // optional
  auto *sequence_lens = context.Input<Tensor>(4);  // [batch_size]
  auto& xshape = X.Shape();
  seq_len_ = xshape[0];
  batch_size_ = xshape[1];
  input_size_ = xshape[2];

  if (sequence_lens == nullptr) {
    vseq_len_ = std::vector<int>(batch_size_, seq_len_);
    seq_len_sum_ = seq_len_ * batch_size_;
    max_seq_len_ = seq_len_;
  } else {
    vseq_len_.resize(batch_size_);
    max_seq_len_ = 0;
    auto ptr = sequence_lens->Data<int32_t>();
    for (int i = 0; i < batch_size_; ++i) {
      vseq_len_[i] = ptr[i];
      if (max_seq_len_ < ptr[i]) {
        max_seq_len_ = ptr[i];
      }
    }
    seq_len_sum_ = batch_size_ * max_seq_len_;
  }

  size_t size_iofc = seq_len_sum_ * hidden_size_ * 4 * sizeof(float);
  out_iofc_->ReAlloc(size_iofc);
  size_t hidden_state_size = sizeof(float) * batch_size_ * hidden_size_;
  hidden_state_->ReAlloc(hidden_state_size);
  hidden_state_->MemSet(0, hidden_state_size);
  cell_state_->ReAlloc(hidden_state_size);
  cell_state_->MemSet(0, hidden_state_size);
  i_->ReAlloc(hidden_state_size);
  o_->ReAlloc(hidden_state_size);
  f_->ReAlloc(hidden_state_size);
  c_->ReAlloc(hidden_state_size);
  if (direction_ != rnn::detail::kForward) {
    input_reverse_->ReAlloc(seq_len_ * batch_size_ * input_size_ * sizeof(float));
    output_reverse_->ReAlloc(seq_len_ * batch_size_ * hidden_size_ * sizeof(float));
  }
  return Status::OK();
}

Status LstmOp::ComputeImplFP32(OpKernelContext &context) const {
  const Tensor &X = *context.Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]
  const Tensor& W = *context.Input<Tensor>(1);  // weights. [num_directions, 4*hidden_size, input_size]
  const Tensor& R = *context.Input<Tensor>(2);  // recurrence weights. [num_directions, 4*hidden_size, hidden_size]
  auto din_ptr = X.Data<float>();
  auto w_ptr = W.Data<float>();
  auto r_ptr = R.Data<float>();
  // optional
  auto *initial_h = context.Input<Tensor>(5);      // initial hidden. [num_directions, batch_size, hidden_size]
  auto *initial_c = context.Input<Tensor>(6);      // initial cell. [num_directions, batch_size, hidden_size]
  auto *P = context.Input<Tensor>(7);              // peephole weights. [num_directions, 3*hidden_size]

  Status status;
//  status = ValidateInputs(X, W, R, B, sequence_lens, initial_h, initial_c, P, batch_size);
//  ORT_RETURN_IF_ERROR(status);

  // LSTM outputs are optional but must be in the same order
  TensorShape Y_dims{seq_len_, num_directions_, batch_size_, hidden_size_};
  Tensor *Y = context.Output(/*index*/ 0, Y_dims);

  TensorShape Y_h_dims{num_directions_, batch_size_, hidden_size_};
  Tensor *Y_h = context.Output(/*index*/ 1, Y_h_dims);

  TensorShape Y_c_dims{num_directions_, batch_size_, hidden_size_};
  Tensor *Y_c = context.Output(/*index*/ 2, Y_c_dims);

  float* output = Y == nullptr? nullptr : Y->MutableData<float>();
  float* output_h = Y_h == nullptr? nullptr : Y_h->MutableData<float>();
  float* output_c = Y_c == nullptr? nullptr : Y_c->MutableData<float>();

  const float* bias_ptr = bias_->Capacity() > 0? bias_->Data<float>() : nullptr;
  const float* peephole_ptr = P == nullptr? nullptr : P->Data<float>();

  // Reset output and return if max sequence length is 0
  if (max_seq_len_ == 0) {
    if (Y != nullptr) std::fill_n(Y->MutableData<float>(), Y_dims.Size(), 0.f);
    if (Y_h != nullptr) std::fill_n(Y_h->MutableData<float>(), Y_h_dims.Size(), 0.f);
    if (Y_c != nullptr) std::fill_n(Y_c->MutableData<float>(), Y_c_dims.Size(), 0.f);
    return Status::OK();
  }
  if (num_directions_ == 1) {
    UniDirectionCompute(num_directions_, direction_,
                        din_ptr, w_ptr, r_ptr,
                        bias_ptr, peephole_ptr,
                        initial_h? initial_h->Data<float>() : nullptr,
                        initial_c? initial_c->Data<float>() : nullptr,
                        act_funcs_,
                        output, output_h, output_c);
  } else {
    // num_directions_ = 2
    auto w_ptr1 = w_ptr;
    auto w_ptr2 = w_ptr + 4 * hidden_size_ * input_size_;
    auto r_ptr1 = r_ptr;
    auto r_ptr2 = r_ptr + 4 * hidden_size_ * hidden_size_;
    const float* init_h1 = nullptr;
    const float* init_h2 = nullptr;
    const float* init_c1 = nullptr;
    const float* init_c2 = nullptr;
    const float* bias_ptr1 = nullptr;
    const float* bias_ptr2 = nullptr;
    const float* peephole_ptr1 = nullptr;
    const float* peephole_ptr2 = nullptr;

    int output_step_size = batch_size_ * hidden_size_;
    auto output1 = output;
    float* output2 = output? output + output_step_size : nullptr;
    auto output_h1 = output_h;
    float* output_h2 = output_h? output_h + output_step_size : nullptr;
    auto output_c1 = output_c;
    float* output_c2 = output_c? output_c + output_step_size : nullptr;

    if (initial_h) {
      init_h1 = initial_h->Data<float>();
      init_h2 = init_h1 + output_step_size;
    }
    if (initial_c) {
      init_c1 = initial_c->Data<float>();
      init_c2 = init_c1 + output_step_size;
    }
    if (bias_ptr) {
      bias_ptr1 = bias_ptr;
      bias_ptr2 = bias_ptr1 + 4 * hidden_size_;
    }
    if (peephole_ptr) {
      peephole_ptr1 = peephole_ptr;
      peephole_ptr2 = peephole_ptr1 + 3 * hidden_size_;
    }
    // forward
    UniDirectionCompute(num_directions_, rnn::detail::kForward,
                        din_ptr, w_ptr1, r_ptr1,
                        bias_ptr1, peephole_ptr1,
                        init_h1, init_c1,
                        act_funcs_,
                        output1, output_h1, output_c1);
    // reverse
    std::vector<ActivationFuncPtr> act_funcs_reverse(act_funcs_.begin() + 3, act_funcs_.end());
    UniDirectionCompute(num_directions_, rnn::detail::kReverse,
                        din_ptr, w_ptr2, r_ptr2,
                        bias_ptr2, peephole_ptr2,
                        init_h2, init_c2,
                        act_funcs_reverse,
                        output2, output_h2, output_c2);
  }
  return Status::OK();
}

Status LstmOp::UniDirectionCompute(int num_directions,
                                   rnn::detail::Direction direction,
                                   const float* x,
                                   const float* wptr,
                                   const float* rptr,
                                   const float* bias_ptr,
                                   const float* peephole_ptr,
                                   const float* init_h,
                                   const float* init_c,
                                   const std::vector<ActivationFuncPtr>& act_funcs,
                                   float* y,
                                   float* y_h,
                                   float* y_c) const {
  float* origin_y = y;
  int step_size = batch_size_ * hidden_size_ * 4;
  int state_step_size = batch_size_ * hidden_size_;
  int y_step_size = state_step_size;
  if (num_directions == 2 && direction == rnn::detail::kForward) {
    y_step_size = 2 * state_step_size;
  }
  if (direction == rnn::detail::kReverse) {
    auto input_reverse_ptr = input_reverse_->MutableData<float>();
    ReverseSequence(x, input_reverse_ptr,
                    vseq_len_, max_seq_len_, batch_size_,
                    input_size_, 1);
    x = input_reverse_ptr;
    if (y) {
      y = output_reverse_->MutableData<float>();
    }
  }
  // x * W
  // preset bias
  int row_len = 4 * hidden_size_;
  float beta = 0.f;
  if (bias_ptr) {
    for (int i = 0; i < seq_len_sum_; ++i) {
      std::memcpy(out_iofc_->MutableData<float>() + i * row_len, bias_ptr, sizeof(float) * row_len);
    }
    beta = 1.f;
  }
  funcs::Sgemm(false, true,
               seq_len_sum_, 4 * hidden_size_, input_size_,
               1.f, x, input_size_,
               wptr, input_size_,
               beta, out_iofc_->MutableData<float>(), 4 * hidden_size_,
               nullptr, false, false, provider_);
  //run through steps sequentially
  for (int step = 0; step < max_seq_len_; step++) {
    // h(t-1) * R
    auto *pre_h_ptr = hidden_state_->Data<float>();
    auto *h_ptr = hidden_state_->MutableData<float>();
    auto *pre_c_ptr = cell_state_->Data<float>();
    auto *c_ptr = cell_state_->MutableData<float>();
    if (step == 0) {
      if (init_h) {
        pre_h_ptr = init_h;
      } else {
        hidden_state_->MemSet(0, state_step_size * sizeof(float));
      }
      if (init_c) {
        pre_c_ptr = init_c;
      } else {
        cell_state_->MemSet(0, state_step_size * sizeof(float));
      }
    }

    float *step_out_ptr = out_iofc_->MutableData<float>() + step * step_size;
    // calculate Xt*(W[iofc]^T) + Ht-1*R[iofc]
    funcs::Sgemm(false, true,
                 batch_size_, 4 * hidden_size_, hidden_size_,
                 1.f, pre_h_ptr, hidden_size_,
                 rptr, hidden_size_,
                 1.f, step_out_ptr, 4 * hidden_size_,
                 nullptr, false, false, provider_);
//    printf("step sgemm:\n");
//    print_data(step_out_ptr, batch_size_ * 4 * hidden_size_, 4 * hidden_size_);
    // gate compute
    GateCompute(step_out_ptr, pre_c_ptr, peephole_ptr, act_funcs, c_ptr, h_ptr);
    if (y) {
      std::memcpy(y + step * y_step_size, h_ptr, sizeof(float) * state_step_size);
    }
#pragma omp parallel for
    for (int i = 0; i < batch_size_; ++i) {
      if (vseq_len_[i] <= step && y) {
        std::memset(y + step * y_step_size + i * hidden_size_, 0, sizeof(float) * hidden_size_);
      }
      if (vseq_len_[i] == step + 1) {
        if (y_h) {
          std::memcpy(y_h + i * hidden_size_, h_ptr + i * hidden_size_, sizeof(float) * hidden_size_);
        }
        if (y_c) {
          std::memcpy(y_c + i * hidden_size_, c_ptr + i * hidden_size_, sizeof(float) * hidden_size_);
        }
      }
    }
  }
  if (y && direction == rnn::detail::kReverse) {
    ReverseSequence(y, origin_y, vseq_len_, seq_len_,
                    batch_size_, hidden_size_, num_directions);
  }
  return Status::OK();
}

Status LstmOp::GateCompute(float* seq_iofc, const float* pre_cell_ptr,
                           const float* peephole,
                           const std::vector<ActivationFuncPtr>& act_funcs,
                           float* cell_ptr, float* hidden_ptr) const {
  // check peephole and clip
  bool clip_done = false;
  if (!peephole && has_clip_ && !input_forget_) {
    GateClip(seq_iofc, seq_iofc, clip_, batch_size_ * 4 * hidden_size_);
    clip_done = true;
  }
  // split to i, o, f, c
  auto iptr = i_->MutableData<float>();
  auto optr = o_->MutableData<float>();
  auto fptr = f_->MutableData<float>();
  auto cptr = c_->MutableData<float>();
  std::vector<float*> iofc_ptr = {iptr, optr, fptr, cptr};
  int compute_size = batch_size_ * hidden_size_;
  funcs::Split(seq_iofc, iofc_ptr, batch_size_, {hidden_size_, hidden_size_, hidden_size_, hidden_size_});

  // check peephole, input gate
  if (peephole) {
#pragma omp parallel for
    for (int i = 0; i < batch_size_; ++i) {
      auto iptr_batch = iptr + i * hidden_size_;
      auto pre_cell_batch = pre_cell_ptr + i * hidden_size_;
      funcs::ElementwiseMac(pre_cell_batch, peephole, iptr_batch, hidden_size_);
    }
  }
  // check clip, input gate, cell
  if (has_clip_ && !clip_done) {
    GateClip(iptr, iptr, clip_, compute_size);
    GateClip(cptr, cptr, clip_, compute_size);
  }

  // do activation, i, c
  act_funcs[0](iptr, iptr, compute_size);
  act_funcs[1](cptr, cptr, compute_size);

  // check input forget, forget gate
  if (input_forget_) {
    InputForget(iptr, fptr, compute_size);
  } else {
    if (peephole) {
#pragma omp parallel for
      for (int i = 0; i < batch_size_; ++i) {
        auto fptr_batch = fptr + i * hidden_size_;
        auto pre_cell_batch = pre_cell_ptr + i * hidden_size_;
        funcs::ElementwiseMac(pre_cell_batch, peephole + 2 * hidden_size_, fptr_batch, hidden_size_);
      }
    }
    if (has_clip_ && !clip_done) {
      GateClip(fptr, fptr, clip_, compute_size);
    }
    // do activation, f
    act_funcs[0](fptr, fptr, compute_size);
  }

//  printf("idata:\n");
//  print_data(iptr, compute_size, hidden_size_);
//  printf("fdata:\n");
//  print_data(fptr, compute_size, hidden_size_);
//  printf("cdata:\n");
//  print_data(cptr, compute_size, hidden_size_);

  // ct_tmp = Ct-1 * ft
  funcs::ElementwiseMul(fptr, pre_cell_ptr, cell_ptr, compute_size);
  // Ct = ct_tmp + ct * it
  funcs::ElementwiseMac(cptr, iptr, cell_ptr, compute_size);
//  printf("CCur:\n");
//  print_data(cell_ptr, compute_size, hidden_size_);

  // check peephole
  if (peephole) {
#pragma omp parallel for
    for (int i = 0; i < batch_size_; ++i) {
      funcs::ElementwiseMac(cell_ptr + i * hidden_size_, peephole + hidden_size_,
                            optr + i * hidden_size_, hidden_size_);
    }
  }
  // check clip
  if (has_clip_ && !clip_done) {
    GateClip(optr, optr, clip_, compute_size);
  }
//  printf("oclipped:\n");
//  print_data(optr, hidden_size_ * batch_size_, hidden_size_);
  // do activation, o
  act_funcs[0](optr, optr, compute_size);
//  printf("odata:\n");
//  print_data(optr, compute_size, hidden_size_);

  // ht_tmp = tanh(Ct)
  act_funcs[2](cell_ptr, hidden_ptr, compute_size);
  // Ht = ot * ht_tmp
  funcs::ElementwiseMul(hidden_ptr, optr, hidden_ptr, compute_size);
//  printf("h:\n");
//  print_data(hidden_ptr, compute_size, hidden_size_);

  return Status::OK();
}

Status LstmOp::PrepareWeights(onnxruntime::OpKernelContext &context) const {
  auto *B = context.Input<Tensor>(3);              // bias. [num_directions, 8*hidden_size]
  if (B != nullptr) {
    bias_->ReAlloc(num_directions_ * 4 * hidden_size_ * sizeof(float));
    // add W bias with R bias
    int offset = 4 * hidden_size_;
    for (int i = 0; i < num_directions_; ++i) {
      auto bias_new_ptr = bias_->MutableData<float>() + i * offset;
      auto w_bias_origin_ptr = B->Data<float>() + i * offset;
      auto r_bias_origin_ptr = w_bias_origin_ptr + offset;
      for (int j = 0; j < offset; ++j) {
        bias_new_ptr[j] = w_bias_origin_ptr[j] + r_bias_origin_ptr[j];
      }
    }
  }
  return Status::OK();
}

}  // namespace arm
}  // namespace onnxruntime
