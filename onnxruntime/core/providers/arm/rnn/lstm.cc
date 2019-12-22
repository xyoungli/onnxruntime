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

Status LstmOp::Compute(OpKernelContext *context) const {
  PrepareWorkspace(*context);
  const Tensor &X = *context->Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]
  Status status;
  // auto& logger = context->Logger();
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
  auto *B = context.Input<Tensor>(3);              // bias. [num_directions, 8*hidden_size]
  auto *sequence_lens = context.Input<Tensor>(4);  // [batch_size]
  auto *P = context.Input<Tensor>(7);              // peephole weights. [num_directions, 3*hidden_size]
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
  size_t hidden_state_size = sizeof(float) * batch_size_ * hidden_size_ * num_directions_;
  hidden_state_->ReAlloc(hidden_state_size);
  hidden_state_->MemSet(0, hidden_state_size);
  cell_state_->ReAlloc(hidden_state_size);
  cell_state_->MemSet(0, hidden_state_size);
  i_->ReAlloc(hidden_state_size);
  o_->ReAlloc(hidden_state_size);
  f_->ReAlloc(hidden_state_size);
  c_->ReAlloc(hidden_state_size);
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

  // Reset output and return if max sequence length is 0
  if (max_seq_len_ == 0) {
    if (Y != nullptr) std::fill_n(Y->MutableData<float>(), Y_dims.Size(), 0.f);
    if (Y_h != nullptr) std::fill_n(Y_h->MutableData<float>(), Y_h_dims.Size(), 0.f);
    if (Y_c != nullptr) std::fill_n(Y_c->MutableData<float>(), Y_c_dims.Size(), 0.f);
    return Status::OK();
  }

  // x * W
  funcs::Sgemm(false, true,
               seq_len_sum_, 4 * hidden_size_, input_size_,
               1.f, din_ptr, input_size_,
               w_ptr, input_size_,
               0.f, out_iofc_->MutableData<float>(), 4 * hidden_size_,
               nullptr, false, false, provider_);

  //run through steps sequentially
  int step_size = batch_size_ * hidden_size_ * 4;
  for (int step = 0; step < max_seq_len_; step++) {
    // h(t-1) * R
    auto *pre_h_ptr = hidden_state_->Data<float>();
    auto *h_ptr = hidden_state_->MutableData<float>();
    auto *pre_c_ptr = cell_state_->Data<float>();
    auto *c_ptr = cell_state_->MutableData<float>();
    if (step == 0 && initial_h) {
      pre_h_ptr = initial_h->Data<float>();
    }
    if (step == 0 && initial_c) {
      pre_c_ptr = initial_c->Data<float>();
    }
    float *step_out_ptr = out_iofc_->MutableData<float>() + step * step_size;
    // calculate Xt*(W[iofc]^T) + Ht-1*R[iofc]
    funcs::Sgemm(false, true,
                 batch_size_, 4 * hidden_size_, hidden_size_,
                 1.f, pre_h_ptr, hidden_size_,
                 r_ptr, hidden_size_,
                 1.f, step_out_ptr, 4 * hidden_size_,
                 nullptr, false, false, provider_);

    // gate compute
//    if (step == max_seq_len_ - 1 && output_h) {
//      h_ptr = output_h;
//    }
//    if (step == max_seq_len_ - 1 && output_c) {
//      c_ptr = output_c;
//    }
    GateCompute(step_out_ptr, pre_c_ptr, c_ptr, h_ptr);
    if (output) {
      std::memcpy(output + step * batch_size_ * hidden_size_, h_ptr, sizeof(float) * batch_size_ * hidden_size_);
    }
    for (int i = 0; i < batch_size_; ++i) {
      if (vseq_len_[i] == step + 1) {
       if (output_h) {
         std::memcpy(output_h + i * hidden_size_, h_ptr + i * hidden_size_, sizeof(float) * hidden_size_);
       }
       if (output_c) {
         std::memcpy(output_c + i * hidden_size_, c_ptr + i * hidden_size_, sizeof(float) * hidden_size_);
       }
      }
    }
  }
  return Status::OK();
}

Status LstmOp::GateCompute(const float* seq_iofc, const float* pre_cell_ptr, float* cell_ptr, float* hiddhen_ptr) const {
  // split to i, o, f, c
  auto iptr = i_->MutableData<float>();
  auto optr = o_->MutableData<float>();
  auto fptr = f_->MutableData<float>();
  auto cptr = c_->MutableData<float>();
  std::vector<float*> iofc_ptr = {iptr, optr,fptr, cptr};
  int compute_size = batch_size_ * hidden_size_;
  funcs::Split(seq_iofc, iofc_ptr, batch_size_, {hidden_size_, hidden_size_, hidden_size_, hidden_size_});
  // do activation
  act_funcs_[0](iptr, iptr, compute_size);
  act_funcs_[0](optr, optr, compute_size);
  act_funcs_[0](fptr, fptr, compute_size);
  act_funcs_[1](cptr, cptr, compute_size);

//  printf("idata:\n");
//  print_data(iptr, compute_size, hidden_size_);
//  printf("odata:\n");
//  print_data(optr, compute_size, hidden_size_);
//  printf("fdata:\n");
//  print_data(fptr, compute_size, hidden_size_);
//  printf("cdata:\n");
//  print_data(cptr, compute_size, hidden_size_);

  // ct_tmp = Ct-1 * ft
  funcs::ElementwiseMul(fptr, pre_cell_ptr, cell_ptr, compute_size);
  // Ct = ct_tmp + ct * it
  funcs::ElementwiseMac(cptr, iptr, cell_ptr, compute_size);
//  printf("c:\n");
//  print_data(cell_ptr, compute_size, hidden_size_);
  // ht_tmp = tanh(Ct)
  act_funcs_[2](cell_ptr, hiddhen_ptr, compute_size);
  // Ht = ot * ht_tmp
  funcs::ElementwiseMul(hiddhen_ptr, optr, hiddhen_ptr, compute_size);
//  printf("h:\n");
//  print_data(hiddhen_ptr, compute_size, hidden_size_);

  return Status::OK();
}

}  // namespace arm
}  // namespace onnxruntime
