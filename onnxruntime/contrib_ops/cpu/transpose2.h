#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/transpose.h"

namespace onnxruntime {
namespace contrib {
class Transpose2 final : public OpKernel, public TransposeBase {
 public:
  Transpose2(const OpKernelInfo& info) : OpKernel(info), TransposeBase(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    // Get input and output:
    const auto* input_tensor_ptr = ctx->Input<Tensor>(0);
    // PTH_ENFORCE(input_tensor_ptr != nullptr);
    const Tensor& X = *input_tensor_ptr;

    const TensorShape& input_shape = X.Shape();
    const std::vector<int64_t>& input_dims = input_shape.GetDims();
    size_t rank = input_dims.size();

    std::vector<int64_t> output_dims(rank);
    const std::vector<size_t>* p_perm;
    std::vector<size_t> default_perm(rank);
    // get the perm from input 1
    const auto* perm_tensor_ptr = ctx->Input<Tensor>(1);
    // const TensorShape& perm_shape = perm_tensor_ptr->Shape();
    // const std::vector<int64_t>& perm_dims = perm_shape.GetDims();
    // PTH_ENFORCE(perm_dims.size() == 1);
    // PTH_ENFORCE(perm_dims[0] == rank);
    const Tensor& perm = *perm_tensor_ptr;
    for (size_t i = 0; i < rank; ++i) {
      const int* perm_data = perm.Data<int>();
      default_perm[i] = perm_data[i];
    }

    const auto& status =
        ComputeOutputShape(X, output_dims, default_perm, p_perm);
    if (!status.IsOK()) return status;

    TensorShape output_shape{output_dims};
    Tensor& Y = *ctx->Output(0, output_shape);

    DoUntypedTranspose(*p_perm, X, Y);

    return Status::OK();
  }
};
}
}
