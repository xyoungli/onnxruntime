#include "transpose.h"
#include <arm_neon.h>

namespace onnxruntime {
namespace arm {
namespace funcs{

void transpose(float *data_out, const float *data_in, int input_rows, int input_cols) {
  int nw = input_cols >> 2;
  int nh = input_rows >> 2;

  float *ptr_out = data_out;
  const float *ptr_in = data_in;
#pragma omp parallel for
  for (int h = 0; h < nh; h++) {
    const float *ptr_din_row = ptr_in + h * 4 * input_cols;
    for (int w = 0; w < nw; w++) {
      float *data_out_ptr = ptr_out + w * 4 * input_rows + h * 4;
      const float *din0 = ptr_din_row;
      const float *din1 = din0 + input_cols;
      const float *din2 = din1 + input_cols;
      const float *din3 = din2 + input_cols;

      float *dout0 = data_out_ptr;
      float *dout1 = dout0 + input_rows;
      float *dout2 = dout1 + input_rows;
      float *dout3 = dout2 + input_rows;
#ifdef __aarch64__
      asm("ldr    q0, [%[in0]]                                \n"
          "ldr    q1, [%[in1]]                                \n"
          "ldr    q2, [%[in2]]                                \n"
          "ldr    q3, [%[in3]]                                \n"
          "trn1   v4.4s, v0.4s, v1.4s                         \n"
          "trn2   v5.4s, v0.4s, v1.4s                         \n"
          "trn1   v6.4s, v2.4s, v3.4s                         \n"
          "trn2   v7.4s, v2.4s, v3.4s                         \n"
          "trn1   v8.2d, v4.2d, v6.2d                         \n"
          "trn1   v9.2d, v5.2d, v7.2d                         \n"
          "trn2   v10.2d, v4.2d, v6.2d                        \n"
          "trn2   v11.2d, v5.2d, v7.2d                        \n"
          "str    q8, [%[out0]]                               \n"
          "str    q9, [%[out1]]                               \n"
          "str   q10, [%[out2]]                               \n"
          "str   q11, [%[out3]]                               \n"
      :
      : [out0] "r"(dout0), [out1] "r"(dout1),
      [out2] "r"(dout2), [out3] "r"(dout3),
      [in0] "r"(din0), [in1] "r"(din1),
      [in2] "r"(din2), [in3] "r"(din3)
      : "v0", "v1", "v2", "v3", "v4", "v5",
              "v6", "v7", "v8", "v9", "v10", "v11");
#else
      asm("vld1.32 {d0, d1}, [%[in0]]    \n"
          "vld1.32 {d2, d3}, [%[in1]]    \n"
          "vld1.32 {d4, d5}, [%[in2]]    \n"
          "vld1.32 {d6, d7}, [%[in3]]    \n"
          "vtrn.32 q0, q1                \n"
          "vtrn.32 q2, q3                \n"
          "vswp d1, d4                   \n"
          "vswp d3, d6                   \n"
          "vst1.32 {d0, d1}, [%[out0]]   \n"
          "vst1.32 {d2, d3}, [%[out1]]   \n"
          "vst1.32 {d4, d5}, [%[out2]]   \n"
          "vst1.32 {d6, d7}, [%[out3]]   \n"
      :
      : [out0] "r"(dout0), [out1] "r"(dout1),
      [out2] "r"(dout2), [out3] "r"(dout3),
      [in0] "r"(din0), [in1] "r"(din1),
      [in2] "r"(din2), [in3] "r"(din3)
      : "q0", "q1", "q2", "q3");
#endif
      ptr_din_row += 4;
    }
  }
  // remian
  for (int h = 0; h < input_rows; h++) {
    for (int w = nw * 4; w < input_cols; w++) {
      const float *data_in_ptr = ptr_in + h * input_cols + w;
      float *data_out_ptr = ptr_out + w * input_rows + h;
      *data_out_ptr = *data_in_ptr;
    }
  }
  for (int w = 0; w < input_cols; w++) {
    for (int h = nh * 4; h < input_rows; h++) {
      const float *data_in_ptr = ptr_in + h * input_cols + w;
      float *data_out_ptr = ptr_out + w * input_rows + h;
      *data_out_ptr = *data_in_ptr;
    }
  }
}

}  // namespace funcs
}  // namespace arm
}  // namespace onnxruntime