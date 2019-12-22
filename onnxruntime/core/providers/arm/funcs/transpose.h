#pragma once

namespace onnxruntime {
namespace arm {
namespace funcs {

void transpose(float *data_out, const float *data_in, int input_rows, int input_cols);

}  // namespace funcs
}  // namespace arm
}  // namespace onnxruntime