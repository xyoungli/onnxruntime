#include "transpose2.h"

#include "core/framework/data_types.h"
#include "core/framework/kernel_def_builder.h"
#include "core/graph/constants.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    Transpose2, kOnnxDomain, 1, kCpuExecutionProvider,
KernelDefBuilder()
.TypeConstraint("T", DataTypeImpl::AllTensorTypes())
.TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
Transpose2);
}
}