// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sgemv.h"
#include <arm_neon.h>

namespace onnxruntime {
namespace arm {
namespace funcs {

void SgemvN(int M,
            int N,
            float alpha,
            const float *A,
            int lda,
            const float *x,
            float beta,
            float *y,
            const float *bias,
            bool with_bias,
            bool with_relu,
            const ARMExecutionProvider *ctx);

void SgemvT(int M,
            int N,
            float alpha,
            const float *A,
            int lda,
            const float *x,
            float beta,
            float *y,
            const float *bias,
            bool with_bias,
            bool with_relu,
            const ARMExecutionProvider *ctx);

#ifdef __aarch64__
#else
void sgemv_trans(const int M,
                 const int N,
                 const float *A,
                 const float *x,
                 float *y,
                 bool flag_bias,
                 const float *bias,
                 bool flag_relu,
                 const ARMExecutionProvider *ctx) {
  int m_cnt8 = M >> 3;
  int m_cnt4 = (M & 7) >> 2;
  int m_remain = M & 7 & 3;
  int ths = ctx->threads();
  int valid_ths = std::min((N + 3) / 4, ths);
  int valid_block = std::max(4, (N / valid_ths + 3) / 4 * 4);
  valid_ths = (N + valid_block - 1) / valid_block;
  int block_cnt = valid_block / 4;
  float zero_buf[M];           // NOLINT
  float y_buf[valid_ths * M];  // NOLINT
  memset(zero_buf, 0, M * sizeof(float));
  if (flag_bias) {
    memcpy(y_buf, bias, M * sizeof(float));
    memset(y_buf + M, 0, (valid_ths - 1) * M * sizeof(float));
  } else {
    memset(y_buf, 0, valid_ths * M * sizeof(float));
  }
#pragma omp parallel for
  for (int t = 0; t < valid_ths; ++t) {
    float *block_y = y_buf + t * M;
    const float *block_x = x + t * valid_block;
    const float *block_A = A + t * valid_block * M;
    for (int i = 0; i < block_cnt; ++i) {
      float *y_ptr = block_y;
      const float *x_ptr = block_x + i * 4;
      const float *in0_ptr = block_A + i * 4 * M;
      const float *in1_ptr = in0_ptr + M;
      const float *in2_ptr = in1_ptr + M;
      const float *in3_ptr = in2_ptr + M;
      int offset = t * valid_block + (i + 1) * 4 - N;
      if (offset > 0) {
        if (offset > 3) {
          in0_ptr = zero_buf;
          in1_ptr = zero_buf;
          in2_ptr = zero_buf;
          in3_ptr = zero_buf;
        } else {
          switch (offset) {
            case 3:
              in1_ptr = zero_buf;
            case 2:
              in2_ptr = zero_buf;
            case 1:
              in3_ptr = zero_buf;
            default:
              break;
          }
        }
      }
      // clang-format off
      if (m_cnt8 > 0) {
        int cnt8 = m_cnt8;
        asm volatile(
            "vld1.32  {d4-d5},  [%[x]]    \n" /* load x   to q2     */
            "vld1.32  {d6-d9},  [%[in0]]! \n" /* load in0 to q3, q4 */
            "vld1.32  {d10-d13},[%[in1]]! \n" /* load in1 to q5, q6 */
            "vld1.32  {d14-d17},[%[in2]]! \n" /* load in2 to q7, q8 */
            "vld1.32  {d18-d21},[%[in3]]! \n" /* load in3 to q9, q10*/
            "1:\n"
            "vld1.32  {d0-d3},  [%[y]]    \n" /*  load y to q0, q1  */
            "vmla.f32 q0, q3,   d4[0]     \n" /*  q0 += q3 * q2[0]  */
            "vmla.f32 q1, q4,   d4[0]     \n" /*  q1 += q4 * q2[0]  */
            "pld  [%[in0]]                \n" /*    preload in0     */
            "vld1.32  {d6-d9},  [%[in0]]! \n" /* load in0 to q3, q4 */
            "vmla.f32 q0, q5,   d4[1]     \n" /*  q0 += q5 * q2[1]  */
            "vmla.f32 q1, q6,   d4[1]     \n" /*  q1 += q6 * q2[1]  */
            "pld  [%[in1]]                \n" /*    preload in1     */
            "vld1.32  {d10-d13},[%[in1]]! \n" /* load in0 to q5, q6 */
            "vmla.f32 q0, q7,   d5[0]     \n" /*  q0 += q7 * q2[2]  */
            "vmla.f32 q1, q8,   d5[0]     \n" /*  q1 += q8 * q2[2]  */
            "pld  [%[in2]]                \n" /*    preload in2     */
            "vld1.32  {d14-d17},[%[in2]]! \n" /* load in0 to q7, q8 */
            "vmla.f32 q0, q9,   d5[1]     \n" /*  q0 += q9 * q2[3]  */
            "vmla.f32 q1, q10,  d5[1]     \n" /*  q1 += q10 * q2[3] */
            "subs %[cnt], %[cnt], #1      \n" /*      sub cnt       */
            "pld  [%[in3]]                \n" /*    preload in3     */
            "vst1.32  {d0-d3},  [%[y]]!   \n" /*  store q0, q1 to y */
            "vld1.32  {d18-d21},[%[in3]]! \n" /* load in0 to q9, q10*/
            "pld  [%[y], #32] \n"             /*     preload y      */
            "bne  1b  \n"                     /*  branch to label 1 */
            "sub  %[in0], %[in0], #32     \n" /* restore in0 address */
            "sub  %[in1], %[in1], #32     \n" /* restore in1 address */
            "sub  %[in2], %[in2], #32     \n" /* restore in2 address */
            "sub  %[in3], %[in3], #32     \n" /* restore in3 address */
            : [cnt] "+r"(cnt8),
              [in0] "+r"(in0_ptr),
              [in1] "+r"(in1_ptr),
              [in2] "+r"(in2_ptr),
              [in3] "+r"(in3_ptr),
              [y] "+r"(y_ptr)
            : [x] "r"(x_ptr)
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", 
              "q7", "q8", "q9", "q10", "cc", "memory"
        );
      }
      if (m_cnt4 > 0) {
        int cnt4 = m_cnt4;
        asm volatile(
            "vld1.32  {d2-d3},  [%[in0]]! \n" /* load in0 to q1  */
            "vld1.32  {d4-d5},  [%[in1]]! \n" /* load in1 to q2  */
            "vld1.32  {d6-d7},  [%[in2]]! \n" /* load in2 to q3  */
            "vld1.32  {d8-d9},  [%[in3]]! \n" /* load in3 to q4  */
            "vld1.32  {d10-d11},[%[x]]    \n" /* load x   to q5  */
            "1:\n"
            "vld1.32  {d0-d1},  [%[y]]    \n" /*   load y to q0    */
            "vmla.f32 q0, q1,   d10[0]    \n" /* q0 += q1 * q5[0]  */
            "pld  [%[in0]]                \n" /*    preload in0    */
            "vld1.32  {d2-d3},  [%[in0]]! \n" /*  load in0 to q1   */
            "vmla.f32 q0, q2,   d10[1]    \n" /* q0 += q2 * q5[1]  */
            "pld  [%[in1]]                \n" /*    preload in1    */
            "vld1.32  {d4-d5},  [%[in1]]! \n" /*  load in0 to q2   */
            "vmla.f32 q0, q3,   d11[0]    \n" /* q0 += q3 * q5[2]  */
            "pld  [%[in2]]                \n" /*    preload in2    */
            "vld1.32  {d6-d7},  [%[in2]]! \n" /*  load in0 to q3   */
            "vmla.f32 q0, q4,   d11[1]    \n" /* q0 += q4 * q5[3]  */
            "subs %[cnt], %[cnt], #1      \n" /*      sub cnt      */
            "pld  [%[in3]]                \n" /*    preload in3    */
            "vst1.32  {d0-d1},  [%[y]]!   \n" /*  store q0 to y    */
            "vld1.32  {d8-d9},  [%[in3]]! \n" /*  load in0 to q4   */
            "bne  1b  \n"                     /*  branch to label 1 */
            "sub  %[in0], %[in0], #16     \n" /* restore in0 address*/
            "sub  %[in1], %[in1], #16     \n" /* restore in1 address*/
            "sub  %[in2], %[in2], #16     \n" /* restore in2 address*/
            "sub  %[in3], %[in3], #16     \n" /* restore in3 address*/
            : [cnt] "+r"(cnt4),
              [in0] "+r"(in0_ptr),
              [in1] "+r"(in1_ptr),
              [in2] "+r"(in2_ptr),
              [in3] "+r"(in3_ptr),
              [y] "+r"(y_ptr)
            : [x] "r"(x_ptr)
            : "q0", "q1", "q2", "q3", "q4", "q5", "cc", "memory"
        );
      }
      // clang-format on
      for (int r = 0; r < m_remain; ++r) {
        float val0 = x_ptr[0] * in0_ptr[r];
        float val1 = x_ptr[1] * in1_ptr[r];
        float val2 = x_ptr[2] * in2_ptr[r];
        float val3 = x_ptr[3] * in3_ptr[r];
        y_ptr[r] += val0 + val1 + val2 + val3;
      }
    }
  }
  //! do reduction
  int rdc_ths = valid_ths >> 1;
  while (rdc_ths > 0) {
#pragma omp parallel for
    for (int t = 0; t < rdc_ths; ++t) {
      float *y0 = y_buf + t * M;
      for (int i = t + rdc_ths; i < valid_ths; i += rdc_ths) {
        float *y0_ptr = y0;
        float *y_ptr = y_buf + i * M;
        for (int j = 0; j < m_cnt8; ++j) {
          float32x4_t val00 = vld1q_f32(y0_ptr + j * 8);
          float32x4_t val01 = vld1q_f32(y0_ptr + j * 8 + 4);
          float32x4_t val10 = vld1q_f32(y_ptr + j * 8);
          float32x4_t val11 = vld1q_f32(y_ptr + j * 8 + 4);
          float32x4_t val0 = vaddq_f32(val00, val10);
          float32x4_t val1 = vaddq_f32(val01, val11);
          vst1q_f32(y0_ptr + j * 8, val0);
          vst1q_f32(y0_ptr + j * 8 + 4, val1);
        }
        y0_ptr += m_cnt8 * 8;
        y_ptr += m_cnt8 * 8;
        for (int j = 0; j < m_cnt4; ++j) {
          float32x4_t val0 = vld1q_f32(y0_ptr + j * 4);
          float32x4_t val1 = vld1q_f32(y_ptr + j * 4);
          float32x4_t val = vaddq_f32(val0, val1);
          vst1q_f32(y0_ptr + j * 4, val);
        }
        y0_ptr += m_cnt4 * 4;
        y_ptr += m_cnt4 * 4;
        for (int j = 0; j < m_remain; ++j) {
          y0_ptr[j] += y_ptr[j];
        }
      }
    }
    valid_ths = rdc_ths;
    rdc_ths = rdc_ths >> 1;
  }
  if (flag_relu) {
    float *in_y = y_buf;
    float32x4_t vzero = vdupq_n_f32(0.f);
    if (m_cnt8 > 0) {
      int cnt8 = m_cnt8;
      asm volatile(
          "vld1.32  {d0-d3},  [%[in_y]]!  \n" /* load y to q0, q1 */
          "1:\n"
          "vmax.f32 q2, q0,   %q[vzero]   \n" /*      q0 relu     */
          "vld1.32  {d0-d1},  [%[in_y]]!  \n" /*   load y to q0   */
          "vmax.f32 q3, q1,   %q[vzero]   \n" /*      q1 relu     */
          "subs %[cnt], %[cnt], #1        \n" /*      sub cnt     */
          "vst1.32  {d4-d7},  [%[out_y]]! \n" /* store q0, q1 to y*/
          "vld1.32  {d2-d3},  [%[in_y]]!  \n" /*   load y to q0   */
          "bne  1b                        \n" /* branch to label 1*/
          "sub  %[in_y],  %[in_y],  #32   \n" /*   restore in_y   */
          : [cnt] "+r"(cnt8), [in_y] "+r"(in_y), [out_y] "+r"(y)
          : [vzero] "w"(vzero)
          : "q0", "q1", "q2", "q3", "cc", "memory");
    }
    if (m_cnt4 > 0) {
      int cnt4 = m_cnt4;
      asm volatile(
          "vld1.32  {d0-d1},  [%[in_y]]!  \n" /*  load y to q0    */
          "1:\n"
          "vmax.f32 q1, q0,   %q[vzero]   \n" /*      q0 relu     */
          "vld1.32  {d0-d1},  [%[in_y]]!  \n" /*   load y to q0   */
          "subs %[cnt], %[cnt], #1        \n" /*      sub cnt     */
          "vst1.32  {d2-d3},  [%[out_y]]! \n" /*  store q1 to y   */
          "bne  1b                        \n" /* branch to label 1*/
          "sub  %[in_y],  %[in_y],  #16   \n" /*   restore in_y   */
          : [cnt] "+r"(cnt4), [in_y] "+r"(in_y), [out_y] "+r"(y)
          : [vzero] "w"(vzero)
          : "q0", "q1", "cc", "memory");
    }
    for (int r = 0; r < m_remain; ++r) {
      y[r] = in_y[r] > 0.f ? in_y[r] : 0.f;
    }
  } else {
    memcpy(y, y_buf, M * sizeof(float));
  }
}
#endif  // __aarch64__

void Sgemv(bool trans,
           int M,
           int N,
           float alpha,
           const float* A,
           int lda,
           const float* x,
           int incx,
           float beta,
           float* y,
           int incy,
           const float* bias,
           bool with_bias,
           bool with_relu,
           const ARMExecutionProvider* ctx) {
  auto alloc_ptr = ctx->GetAllocator(0, OrtMemTypeDefault);
  auto x_ptr = x;
  auto y_ptr = y;
  float* x_tmp = nullptr;
  float* y_tmp = nullptr;
  if (incx > 1) {
    x_tmp = static_cast<float*>(
            alloc_ptr->Alloc(N * sizeof(float)));
    for (int i = 0; i < N; ++i) {
      x_tmp[i] = x[i * incx];
    }
    x_ptr = x_tmp;
  }
  if (incy > 1) {
    y_tmp = static_cast<float*>(
            alloc_ptr->Alloc(M * sizeof(float)));
    for (int i = 0; i < M; ++i) {
      y_tmp[i] = y[i * incy];
    }
    y_ptr = y_tmp;
  }
  if (trans) {
    SgemvT(M, N, alpha, A, lda, x_ptr, beta, y_ptr, bias, with_bias, with_relu, ctx);
  } else {
    SgemvN(M, N, alpha, A, lda, x_ptr, beta, y_ptr, bias, with_bias, with_relu, ctx);
  }
  if (incy > 1) {
    for (int i = 0; i < M; ++i) {
      y[i * incy] = y_ptr[i];
    }
  }
  if (x_tmp) {
    alloc_ptr->Free(x_tmp);
  }
  if (y_tmp) {
    alloc_ptr->Free(y_tmp);
  }
}

void SgemvN(int M,
            int N,
            float alpha,
            const float *A,
            int lda,
            const float *x,
            float beta,
            float *y,
            const float *bias,
            bool with_bias,
            bool with_relu,
            const ARMExecutionProvider *ctx) {
  (void)(ctx);
  int cnt = N >> 3;
  int tail = N & 7;
  uint32_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint32x4_t vmask1 =
          vcltq_u32(vld1q_u32(mask_buffer), vdupq_n_u32(tail));
  uint32x4_t vmask2 =
          vcltq_u32(vld1q_u32(mask_buffer + 4), vdupq_n_u32(tail));
#ifdef __aarch64__
  float32x4_t valpha = vdupq_n_f32(alpha);
  float32x4_t vbeta = vdupq_n_f32(beta);
  int out_cnt = M >> 3;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 8;
    float *ptr_out = y + out_idx;
    const float *ptr_in = x;
    const float *ptr_w0 = A + (lda * out_idx);
    const float *ptr_w1 = ptr_w0 + lda;
    const float *ptr_w2 = ptr_w1 + lda;
    const float *ptr_w3 = ptr_w2 + lda;
    const float *ptr_w4 = ptr_w3 + lda;
    const float *ptr_w5 = ptr_w4 + lda;
    const float *ptr_w6 = ptr_w5 + lda;
    const float *ptr_w7 = ptr_w6 + lda;
    const float *bias_ptr = bias + out_idx;
    int cnt_loop = cnt;
    asm volatile(
    "prfm  pldl1keep, [%[in]]   \n" /* preload din */
    "prfm  pldl1keep, [%[w0]]   \n" /* preload w0 */
    "prfm  pldl1keep, [%[w1]]   \n" /* preload w1 */
    "prfm  pldl1keep, [%[w2]]   \n" /* preload w2 */
    "prfm  pldl1keep, [%[w3]]   \n" /* preload w3 */
    "prfm  pldl1keep, [%[w4]]   \n" /* preload w4 */
    "prfm  pldl1keep, [%[w5]]   \n" /* preload w5 */
    "prfm  pldl1keep, [%[w6]]   \n" /* preload w6 */
    "prfm  pldl1keep, [%[w7]]   \n" /* preload w7 */
    "movi   v0.4s,  #0          \n" /* set out0 to 0 */
    "movi   v1.4s,  #0          \n" /* set out1 to 0 */
    "movi   v2.4s,  #0          \n" /* set out2 to 0 */
    "movi   v3.4s,  #0          \n" /* set out3 to 0 */
    "ldr q8, [%[in]], #16       \n" /* load input 4 float */
    "ldp q10, q11, [%[w0]], #32 \n" /* load w0 8 float */
    "movi   v4.4s,  #0          \n" /* set out4 to 0 */
    "movi   v5.4s,  #0          \n" /* set out5 to 0 */
    "ldp q12, q13, [%[w1]], #32 \n" /* load w1 8 float */
    "movi   v6.4s,  #0          \n" /* set out6 to 0 */
    "movi   v7.4s,  #0          \n" /* set out7 to 0 */
    "ldp q14, q15, [%[w2]], #32 \n" /* load w2 8 float */
    "ldp q16, q17, [%[w3]], #32 \n" /* load w3 8 float */
    /* check main loop */
    "cbz %w[cnt], 0f            \n" /* check loop */
    "1:                         \n" /* main loop */
    "ldp q18, q19, [%[w4]], #32 \n" /* load w4 8 float */
    "ldr q9, [%[in]], #16       \n" /* load input 4 float */
    "fmla v0.4s, v8.4s, v10.4s  \n" /* mul + add*/
    "fmla v1.4s, v8.4s, v12.4s  \n" /* mul + add*/
    "ldp q20, q21, [%[w5]], #32 \n" /* load w5 8 float */
    "fmla v2.4s, v8.4s, v14.4s  \n" /* mul + add*/
    "fmla v3.4s, v8.4s, v16.4s  \n" /* mul + add*/
    "ldp q22, q23, [%[w6]], #32 \n" /* load w6 8 float */
    "fmla v4.4s, v8.4s, v18.4s  \n" /* mul + add*/
    "fmla v5.4s, v8.4s, v20.4s  \n" /* mul + add*/
    "ldp q24, q25, [%[w7]], #32 \n" /* load w7 8 float */
    "fmla v6.4s, v8.4s, v22.4s  \n" /* mul + add*/
    "fmla v7.4s, v8.4s, v24.4s  \n" /* mul + add*/
    "ldr q8, [%[in]], #16       \n" /* load input 4 float */
    "fmla v0.4s, v9.4s, v11.4s  \n" /* mul + add*/
    "fmla v1.4s, v9.4s, v13.4s  \n" /* mul + add*/
    "ldp q10, q11, [%[w0]], #32 \n" /* load w0 8 float */
    "subs %w[cnt], %w[cnt], #1  \n" /* sub main loop count */
    "fmla v2.4s, v9.4s, v15.4s  \n" /* mul + add*/
    "fmla v3.4s, v9.4s, v17.4s  \n" /* mul + add*/
    "ldp q12, q13, [%[w1]], #32 \n" /* load w1 8 float */
    "fmla v4.4s, v9.4s, v19.4s  \n" /* mul + add*/
    "fmla v5.4s, v9.4s, v21.4s  \n" /* mul + add*/
    "ldp q14, q15, [%[w2]], #32 \n" /* load w2 8 float */
    "fmla v6.4s, v9.4s, v23.4s  \n" /* mul + add*/
    "fmla v7.4s, v9.4s, v25.4s  \n" /* mul + add*/
    "ldp q16, q17, [%[w3]], #32 \n" /* load w3 8 float */
    "bne 1b                     \n" /* jump to main loop */
    "0:                         \n" /* tail */
    "cbz %w[tail], 2f           \n"  /* check tail */
    "movi   v31.4s,  #0         \n" /* set padding to 0 */
    "ldp q18, q19, [%[w4]]      \n" /* load w4 8 float */
    "bif  v8.16b, v31.16b, %[mask1].16b\n" /* pad 0 */
    "bif  v10.16b, v31.16b, %[mask1].16b\n" /* pad 0 */
    "bif  v12.16b, v31.16b, %[mask1].16b\n" /* pad 0 */
    "ldp q20, q21, [%[w5]]      \n" /* load w5 8 float */
    "ldr q9, [%[in]], #16       \n" /* load input 4 float */
    "bif  v14.16b, v31.16b, %[mask1].16b\n" /* pad 0 */
    "bif  v16.16b, v31.16b, %[mask1].16b\n" /* pad 0 */
    "fmla v0.4s, v8.4s, v10.4s  \n" /* mul + add*/
    "fmla v1.4s, v8.4s, v12.4s  \n" /* mul + add*/
    "bif  v18.16b, v31.16b, %[mask1].16b\n" /* pad 0 */
    "bif  v20.16b, v31.16b, %[mask1].16b\n" /* pad 0 */
    "ldp q22, q23, [%[w6]]      \n" /* load w6 8 float */
    "ldp q24, q25, [%[w7]]      \n" /* load w7 8 float */
    "fmla v2.4s, v8.4s, v14.4s  \n" /* mul + add*/
    "fmla v3.4s, v8.4s, v16.4s  \n" /* mul + add*/
    "bif  v22.16b, v31.16b, %[mask1].16b\n" /* pad 0 */
    "bif  v24.16b, v31.16b, %[mask1].16b\n" /* pad 0 */
    "fmla v4.4s, v8.4s, v18.4s  \n" /* mul + add*/
    "fmla v5.4s, v8.4s, v20.4s  \n" /* mul + add*/
    "bif  v9.16b, v31.16b, %[mask2].16b\n" /* pad 0 */
    "bif  v11.16b, v31.16b, %[mask2].16b\n" /* pad 0 */
    "bif  v13.16b, v31.16b, %[mask2].16b\n" /* pad 0 */
    "fmla v6.4s, v8.4s, v22.4s  \n" /* mul + add*/
    "fmla v7.4s, v8.4s, v24.4s  \n" /* mul + add*/
    "bif  v15.16b, v31.16b, %[mask2].16b\n" /* pad 0 */
    "bif  v17.16b, v31.16b, %[mask2].16b\n" /* pad 0 */
    "fmla v0.4s, v9.4s, v11.4s  \n" /* mul + add*/
    "fmla v1.4s, v9.4s, v13.4s  \n" /* mul + add*/
    "bif  v19.16b, v31.16b, %[mask2].16b\n" /* pad 0 */
    "bif  v21.16b, v31.16b, %[mask2].16b\n" /* pad 0 */
    "fmla v2.4s, v9.4s, v15.4s  \n" /* mul + add*/
    "fmla v3.4s, v9.4s, v17.4s  \n" /* mul + add*/
    "bif  v23.16b, v31.16b, %[mask2].16b\n" /* pad 0 */
    "bif  v25.16b, v31.16b, %[mask2].16b\n" /* pad 0 */
    "fmla v4.4s, v9.4s, v19.4s  \n" /* mul + add*/
    "fmla v5.4s, v9.4s, v21.4s  \n" /* mul + add*/
    "fmla v6.4s, v9.4s, v23.4s  \n" /* mul + add*/
    "fmla v7.4s, v9.4s, v25.4s  \n" /* mul + add*/
    /* pair add to final result */
    "2:                         \n"  /* reduce to scale */
    "faddp  v16.4s, v0.4s, v1.4s\n"  /* pair add to vector */
    "faddp  v17.4s, v2.4s, v3.4s\n"  /* pair add to vector */
    "faddp  v18.4s, v4.4s, v5.4s\n"  /* pair add to vector */
    "faddp  v19.4s, v6.4s, v7.4s\n"  /* pair add to vector */
    "faddp  v8.4s,v16.4s,v17.4s \n"  /* pair add to vector */
    "faddp  v9.4s,v18.4s,v19.4s \n"  /* pair add to vector */
    /* deal with alpha and beta */
    "ldp q0, q1, [%[out]]       \n"  /* load out to q0, q1 */
    "movi v2.4s, #0             \n"
    "movi v3.4s, #0             \n"
    "cbz  %w[bias], 3f          \n" /* check bias */
    "ldp   q2, q3, [%[bias_ptr]]\n" /* load bias to q2, q3*/
    "3:                         \n"
    "fmla   v2.4s, v0.4s, %[beta].4s\n" /* y = c * beta + bias */
    "fmla   v3.4s, v1.4s, %[beta].4s\n" /* y = c * beta + bias */
    "fmla   v2.4s, v8.4s, %[alpha].4s\n" /* y = y + alpha * Ax */
    "fmla   v3.4s, v9.4s, %[alpha].4s\n" /* y = y + alpha * Ax */
    "cbz    %w[relu], 4f        \n" /* check relu */
    "movi   v31.4s,  #0         \n" /* set padding to 0 */
    "fmax   v2.4s, v2.4s, v31.4s\n" /* relu */
    "fmax   v3.4s, v3.4s, v31.4s\n" /* relu */
    "4:                         \n" /* save result */
    "stp q2, q3, [%[out]]\n"
    : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0),
    [w1] "+r"(ptr_w1), [w2] "+r"(ptr_w2),
    [w3] "+r"(ptr_w3), [w4] "+r"(ptr_w4),
    [w5] "+r"(ptr_w5), [w6] "+r"(ptr_w6),
    [w7] "+r"(ptr_w7), [cnt] "+r"(cnt_loop)
    : [out] "r"(ptr_out), [bias_ptr] "r"(bias_ptr),
      [bias] "r"(with_bias), [relu] "r"(with_relu), [tail] "r"(tail),
      [alpha] "w"(valpha), [beta] "w"(vbeta),
      [mask1]"w"(vmask1), [mask2]"w"(vmask2)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
      "v10", "v11", "v12", "v13", "v14", "v15",
      "v16", "v17", "v18", "v19", "v20", "v21",
      "v22", "v23", "v24", "v25", "v31", "cc", "memory");
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 8; j < M; ++j) {
    float *ptr_out = y + j;
    const float *ptr_in = x;
    const float *ptr_w0 = A + (lda * j);
    int cnt_loop = cnt;
    float bias0 = bias[j];
    asm volatile(
    "prfm  pldl1keep, [%[in]]   \n" /* preload din */
    "prfm  pldl1keep, [%[w0]]   \n" /* preload w0 */
    "ldp q8, q9, [%[in]], #32   \n" /* load input 8 float */
    "ldp q10, q11, [%[w0]], #32 \n" /* load w0 8 float */
    "movi   v0.4s,  #0          \n" /* set out0 to 0 */
    "movi   v1.4s,  #0          \n" /* set out0 to 0 */
    /* check main loop */
    "cbz %w[cnt], 0f            \n" /* check loop */
    "1:                         \n" /* main loop */
    "fmla v0.4s, v8.4s, v10.4s  \n" /* mul + add*/
    "fmla v1.4s, v9.4s, v11.4s  \n" /* mul + add*/
    "subs %w[cnt], %w[cnt], #1  \n" /* sub main loop count */
    "ldp q8, q9, [%[in]], #32   \n" /* load input 8 float */
    "ldp q10, q11, [%[w0]], #32 \n" /* load w0 8 float */
    "bne 1b                     \n" /* jump to main loop */
    "0:                         \n" /* tail */
    "cbz %w[tail], 2f           \n" /* check tail */
    "movi   v31.4s,  #0         \n" /* set padding to 0 */
    "bif  v8.16b, v31.16b, %[mask1].16b\n" /* pad 0 */
    "bif  v10.16b, v31.16b, %[mask1].16b\n" /* pad 0 */
    "bif  v9.16b, v31.16b, %[mask2].16b\n" /* pad 0 */
    "bif  v11.16b, v31.16b, %[mask2].16b\n" /* pad 0 */
    "fmla v0.4s, v8.4s, v10.4s  \n" /* mul + add*/
    "fmla v1.4s, v9.4s, v11.4s  \n" /* mul + add*/
    /* pair add to final result */
    "2:                         \n" /* reduce to scale */
    "faddp  v2.4s, v0.4s, v1.4s \n" /* pair add to vector */
    "faddp  v3.4s, v2.4s, v2.4s \n" /* pair add to vector */
    "faddp  s0, v3.2s           \n" /* pair add to scale */
    /* end */
    "fmov s1, %w[alpha]         \n"
    "fmov s2, %w[beta]          \n"
    "ldr    s3, [%[out]]        \n" /* load out, 1 float */
    "movi   d4, #0              \n"
    "cbz  %w[bias], 3f          \n" /* check bias */
    "fmov   s4,  %w[bias0]      \n" /* set out0 = bias0 */
    "3:                         \n"
    "fmadd s5, s2, s3, s4       \n" /* y = c * beta + bias */
    "fmadd s6, s0, s1, s5       \n" /* y = alpha * Ax + y */
    "cbz  %w[relu], 4f          \n" /* check relu */
    "movi   d4, #0              \n" /* zero data for relu */
    "fmax   s6, s6, s4          \n" /* relu */
    "4:                         \n" /* end */
    "str s6, [%[out]]           \n" /* save result */
    : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0),
    [cnt] "+r"(cnt_loop)
    : [out] "r"(ptr_out), [bias0] "r"(bias0),
    [bias] "r"(with_bias), [relu] "r"(with_relu),
    [alpha] "r"(alpha), [beta] "r"(beta), [tail] "r"(tail),
    [mask1]"w"(vmask1), [mask2]"w"(vmask2)
    : "v0", "v1", "v8", "v9", "v10", "v11", "v16", "v17", "v31", "cc", "memory");
  }
#else  // __aarch64__
  int out_cnt = M >> 2;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 4;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;
    float bias0 = bias[out_idx];
    float bias1 = bias[out_idx + 1];
    float bias2 = bias[out_idx + 2];
    float bias3 = bias[out_idx + 3];

    int cnt_loop = cnt;
    int tail_loop = tail;
    asm volatile(SGEMV_IN_4_BIAS SGEMV_KERNEL_4 SGEMV_OUT_4_RELU
                 : [in] "+r"(ptr_in),
                   [w0] "+r"(ptr_w0),
                   [w1] "+r"(ptr_w1),
                   [w2] "+r"(ptr_w2),
                   [w3] "+r"(ptr_w3),
                   [cnt] "+r"(cnt_loop),
                   [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out),
                   [bias0] "r"(bias0),
                   [bias1] "r"(bias1),
                   [bias2] "r"(bias2),
                   [bias3] "r"(bias3)
                 : "q0",
                   "q1",
                   "q2",
                   "q3",
                   "q4",
                   "q5",
                   "q6",
                   "q7",
                   "q8",
                   "q9",
                   "q10",
                   "q11",
                   "q12",
                   "q13",
                   "cc",
                   "memory");
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 4; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    float bias0 = bias[j];
    asm volatile(SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_RELU
                 : [in] "+r"(ptr_in),
                   [w0] "+r"(ptr_w0),
                   [cnt] "+r"(cnt_loop),
                   [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out), [bias0] "r"(bias0)
                 : "q0", "q1", "q12", "q13", "q14", "q15", "cc", "memory");
  }
#endif  // __aarch64__
}

void SgemvT(int M,
            int N,
            float alpha,
            const float *A,
            int lda,
            const float *x,
            float beta,
            float *y,
            const float *bias,
            bool with_bias,
            bool with_relu,
            const ARMExecutionProvider *ctx) {
  int m_cnt16 = M >> 4;
  int m_cnt8 = (M & 15) >> 3;
  int m_cnt4 = (M & 7) >> 2;
  int m_remain = M & 3;
  int ths = ctx->Threads();
  int valid_ths = std::min((N + 3) / 4, ths);
  int valid_block = std::max(4, (N / valid_ths + 3) / 4 * 4);
  valid_ths = (N + valid_block - 1) / valid_block;
  int block_cnt = valid_block / 4;
  float zero_buf[M];
  float y_buf[valid_ths * M];
  memset(zero_buf, 0, M * sizeof(float));
  memset(y_buf, 0, valid_ths * M * sizeof(float));
#pragma omp parallel for
  for (int t = 0; t < valid_ths; ++t) {
    float *block_y = y_buf + t * M;
    const float *block_x = x + t * valid_block;
    const float *block_A = A + t * valid_block * lda;
    for (int i = 0; i < block_cnt; ++i) {
      float *y_ptr = block_y;
      const float *x_ptr = block_x + i * 4;
      const float *in0_ptr = block_A + i * 4 * lda;
      const float *in1_ptr = in0_ptr + lda;
      const float *in2_ptr = in1_ptr + lda;
      const float *in3_ptr = in2_ptr + lda;
      int offset = t * valid_block + (i + 1) * 4 - N;
      if (offset > 0) {
        if (offset > 3) {
          in0_ptr = zero_buf;
          in1_ptr = zero_buf;
          in2_ptr = zero_buf;
          in3_ptr = zero_buf;
        } else {
          switch (offset) {
            case 3:
              in1_ptr = zero_buf;
            case 2:
              in2_ptr = zero_buf;
            case 1:
              in3_ptr = zero_buf;
            default:
              break;
          }
        }
      }
      if (m_cnt16 > 0) {
        int cnt16 = m_cnt16;
        asm volatile(
        "ld1  {v4.4s},  [%[x]]    \n"                               /* load x   to v4     */
        "ld1  {v5.4s,  v6.4s,  v7.4s,  v8.4s},   [%[in0]], #64 \n"  /* load in0 to v5,  v6,  v7,  v8  */
        "ld1  {v9.4s,  v10.4s, v11.4s, v12.4s},  [%[in1]], #64 \n"  /* load in1 to v9,  v10, v11, v12 */
        "ld1  {v13.4s, v14.4s, v15.4s, v16.4s},  [%[in2]], #64 \n"  /* load in2 to v13, v14, v15, v16 */
        "ld1  {v17.4s, v18.4s, v19.4s, v20.4s},  [%[in3]], #64 \n"  /* load in3 to v17, v18, v19, v20 */
        "1:\n"
        "ld1  {v0.4s, v1.4s, v2.4s, v3.4s},  [%[y]]    \n"        /*load y to v0, v1, v2, v3  */
        "fmla v0.4s,  v5.4s,  v4.s[0]     \n" /*  v0 += v5 * v4[0]  */
        "fmla v1.4s,  v6.4s,  v4.s[0]     \n" /*  v1 += v6 * v4[0]  */
        "fmla v2.4s,  v7.4s,  v4.s[0]     \n" /*  v2 += v7 * v4[0]  */
        "fmla v3.4s,  v8.4s,  v4.s[0]     \n" /*  v3 += v8 * v4[0]  */
        "ld1  {v5.4s, v6.4s,  v7.4s,  v8.4s},   [%[in0]], #64 \n" /* load in0 to v5,  v6,  v7,  v8  */
        "fmla v0.4s,  v9.4s,  v4.s[1]     \n" /*  v0 += v9  * v4[1]  */
        "fmla v1.4s,  v10.4s, v4.s[1]     \n" /*  v1 += v10 * v4[1]  */
        "fmla v2.4s,  v11.4s, v4.s[1]     \n" /*  v2 += v11 * v4[1]  */
        "fmla v3.4s,  v12.4s, v4.s[1]     \n" /*  v3 += v12 * v4[1]  */
        "ld1  {v9.4s, v10.4s, v11.4s, v12.4s},  [%[in1]], #64 \n" /* load in1 to v9,  v10, v11, v12 */
        "fmla v0.4s,  v13.4s, v4.s[2]     \n" /*  v0 += v13 * v4[2]  */
        "fmla v1.4s,  v14.4s, v4.s[2]     \n" /*  v1 += v14 * v4[2]  */
        "fmla v2.4s,  v15.4s, v4.s[2]     \n" /*  v2 += v15 * v4[2]  */
        "fmla v3.4s,  v16.4s, v4.s[2]     \n" /*  v3 += v16 * v4[2]  */
        "ld1  {v13.4s, v14.4s, v15.4s, v16.4s}, [%[in2]], #64 \n" /* load in2 to v13, v14, v15, v16 */
        "fmla v0.4s,  v17.4s, v4.s[3]     \n" /*  v0 += v17 * v4[3]  */
        "fmla v1.4s,  v18.4s, v4.s[3]     \n" /*  v1 += v18 * v4[3]  */
        "fmla v2.4s,  v19.4s, v4.s[3]     \n" /*  v2 += v19 * v4[3]  */
        "fmla v3.4s,  v20.4s, v4.s[3]     \n" /*  v3 += v20 * v4[3]  */
        "ld1  {v17.4s, v18.4s, v19.4s, v20.4s}, [%[in3]], #64 \n" /* load in3 to v17, v18, v19, v20 */
        "subs %w[cnt], %w[cnt], #1        \n" /*       sub cnt       */
        "st1  {v0.4s, v1.4s, v2.4s, v3.4s}, [%[y]], #64   \n"     /*  store v0, v1, v2, v3 to y */
        "bne  1b  \n"                     /*  branch to label 1 */
        "sub  %[in0], %[in0], #64     \n" /* restore in0 address */
        "sub  %[in1], %[in1], #64     \n" /* restore in1 address */
        "sub  %[in2], %[in2], #64     \n" /* restore in2 address */
        "sub  %[in3], %[in3], #64     \n" /* restore in3 address */
        : [cnt] "+r"(cnt16),
        [in0] "+r"(in0_ptr),
        [in1] "+r"(in1_ptr),
        [in2] "+r"(in2_ptr),
        [in3] "+r"(in3_ptr),
        [y] "+r"(y_ptr)
        : [x] "r"(x_ptr)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
                "v17", "v18", "v19", "v20", "cc", "memory"
        );
      }
      if (m_cnt8 > 0) {
        int cnt8 = m_cnt8;
        asm volatile(
        "ld1  {v2.4s},  [%[x]]                \n" /* load x   to v2     */
        "ld1  {v3.4s, v4.4s},  [%[in0]], #32  \n" /* load in0 to v3, v4 */
        "ld1  {v5.4s, v6.4s},  [%[in1]], #32  \n" /* load in1 to v5, v6 */
        "ld1  {v7.4s, v8.4s},  [%[in2]], #32  \n" /* load in2 to v7, v8 */
        "ld1  {v9.4s, v10.4s}, [%[in3]], #32  \n" /* load in3 to v9, v10*/
        "1:\n"
        "ld1  {v0.4s, v1.4s}, [%[y]]    \n" /*  load y to v0, v1  */
        "fmla v0.4s, v3.4s,   v2.s[0]   \n" /*  v0 += v3 * v2[0]  */
        "fmla v1.4s, v4.4s,   v2.s[0]   \n" /*  v1 += v4 * v2[0]  */
        "prfm pldl1keep,      [%[in0]]  \n" /*    preload in0     */
        "ld1  {v3.4s, v4.4s}, [%[in0]], #32 \n" /* load in0 to v3, v4 */
        "fmla v0.4s, v5.4s,   v2.s[1]   \n" /*  v0 += v5 * v2[1]  */
        "fmla v1.4s, v6.4s,   v2.s[1]   \n" /*  v1 += v6 * v2[1]  */
        "prfm pldl1keep,      [%[in1]]  \n" /*    preload in1     */
        "ld1  {v5.4s, v6.4s}, [%[in1]], #32 \n" /* load in0 to v5, v6 */
        "fmla v0.4s, v7.4s,   v2.s[2]   \n" /*  v0 += v7 * v2[2]  */
        "fmla v1.4s, v8.4s,   v2.s[2]   \n" /*  v1 += v8 * v2[2]  */
        "prfm pldl1keep,      [%[in2]]  \n" /*    preload in2     */
        "ld1  {v7.4s, v8.4s}, [%[in2]], #32 \n" /* load in0 to v7, v8 */
        "fmla v0.4s, v9.4s,   v2.s[3]   \n" /*  v0 += v9 * v2[3]  */
        "fmla v1.4s, v10.4s,  v2.s[3]   \n" /*  v1 += v10 * v2[3] */
        "subs %w[cnt], %w[cnt], #1      \n" /*      sub cnt       */
        "prfm pldl1keep,      [%[in3]]  \n" /*    preload in3     */
        "st1  {v0.4s, v1.4s}, [%[y]],   #32 \n" /*  store v0, v1 to y */
        "ld1  {v9.4s, v10.4s},[%[in3]], #32 \n" /* load in0 to v9, v10*/
        "bne  1b  \n"                       /*  branch to label 1 */
        "sub  %[in0], %[in0], #32     \n" /* restore in0 address */
        "sub  %[in1], %[in1], #32     \n" /* restore in1 address */
        "sub  %[in2], %[in2], #32     \n" /* restore in2 address */
        "sub  %[in3], %[in3], #32     \n" /* restore in3 address */
        : [cnt] "+r"(cnt8), [y] "+r"(y_ptr),
        [in0] "+r"(in0_ptr), [in1] "+r"(in1_ptr),
        [in2] "+r"(in2_ptr), [in3] "+r"(in3_ptr)
        : [x] "r"(x_ptr)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                "v7", "v8", "v9", "v10", "cc", "memory"
        );
      }
      if (m_cnt4 > 0) {
        int cnt4 = m_cnt4;
        asm volatile(
        "ld1  {v1.4s},  [%[in0]], #16 \n" /* load in0 to v1  */
        "ld1  {v2.4s},  [%[in1]], #16 \n" /* load in1 to v2  */
        "ld1  {v3.4s},  [%[in2]], #16 \n" /* load in2 to v3  */
        "ld1  {v4.4s},  [%[in3]], #16 \n" /* load in3 to v4  */
        "ld1  {v5.4s},  [%[x]]        \n" /* load x   to v5  */
        "1:\n"
        "ld1  {v0.4s},  [%[y]]        \n" /*   load y to v0    */
        "fmla v0.4s, v1.4s, v5.s[0]   \n" /* v0 += v1 * v5[0]  */
        "prfm  pldl1keep,   [%[in0]]  \n" /*    preload in0    */
        "ld1  {v1.4s},  [%[in0]], #16 \n" /*  load in0 to v1   */
        "fmla v0.4s, v2.4s, v5.s[1]   \n" /* v0 += v2 * v5[1]  */
        "prfm  pldl1keep,  [%[in1]]   \n" /*    preload in1    */
        "ld1  {v2.4s},  [%[in1]], #16 \n" /*  load in1 to v2   */
        "fmla v0.4s, v3.4s, v5.s[2]   \n" /* v0 += v3 * v5[2]  */
        "prfm pldl1keep,  [%[in2]]    \n" /*    preload in2    */
        "ld1  {v3.4s},  [%[in2]], #16 \n" /*  load in2 to v3   */
        "fmla v0.4s, v4.4s, v5.s[3]   \n" /* v0 += v4 * v5[3]  */
        "subs %w[cnt], %w[cnt], #1    \n" /*      sub cnt      */
        "prfm pldl1keep,  [%[in3]]    \n" /*    preload in3    */
        "st1  {v0.4s},  [%[y]], #16   \n" /*  store v0 to y    */
        "ld1  {v4.4s},  [%[in3]], #16 \n" /*  load in3 to v4   */
        "bne  1b  \n"                     /* branch to label 1 */
        "sub  %[in0], %[in0], #16     \n" /* restore in0 address*/
        "sub  %[in1], %[in1], #16     \n" /* restore in1 address*/
        "sub  %[in2], %[in2], #16     \n" /* restore in2 address*/
        "sub  %[in3], %[in3], #16     \n" /* restore in3 address*/
        : [cnt] "+r"(cnt4), [y] "+r"(y_ptr),
        [in0] "+r"(in0_ptr), [in1] "+r"(in1_ptr),
        [in2] "+r"(in2_ptr), [in3] "+r"(in3_ptr)
        : [x] "r"(x_ptr)
        : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory"
        );
      }
      for (int r = 0; r < m_remain; ++r) {
        float val0 = x_ptr[0] * in0_ptr[r];
        float val1 = x_ptr[1] * in1_ptr[r];
        float val2 = x_ptr[2] * in2_ptr[r];
        float val3 = x_ptr[3] * in3_ptr[r];
        y_ptr[r] += val0 + val1 + val2 + val3;
      }
    }
  }
  int cnt4 = M >> 2;
  int remain = M & 3;
  //! do reduction
  int rdc_ths = valid_ths >> 1;
  while (rdc_ths > 0) {
#pragma omp parallel for
    for (int t = 0; t < rdc_ths; ++t) {
      float *y0 = y_buf + t * M;
      for (int i = t + rdc_ths; i < valid_ths; i += rdc_ths) {
        float *y0_ptr = y0;
        float *y_ptr = y_buf + i * M;
        for (int j = 0; j < cnt4; ++j) {
          float32x4_t val0 = vld1q_f32(y0_ptr + j * 4);
          float32x4_t val1 = vld1q_f32(y_ptr + j * 4);
          float32x4_t val = vaddq_f32(val0, val1);
          vst1q_f32(y0_ptr + j * 4, val);
        }
        y0_ptr += cnt4 * 4;
        y_ptr += cnt4 * 4;
        for (int j = 0; j < remain; ++j) {
          y0_ptr[j] += y_ptr[j];
        }
      }
    }
    valid_ths = rdc_ths;
    rdc_ths = rdc_ths >> 1;
  }

  // deal with alpha and beta
  auto y_buf_ptr = y_buf;
  if (cnt4 > 0) {
    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t vbeta = vdupq_n_f32(beta);
    float32x4_t vzero = vdupq_n_f32(0.f);
    asm volatile(
    "ld1  {v2.4s},  [%[out_y]]      \n" /*  load y to v2    */
    "ld1  {v0.4s},  [%[in_y]], #16  \n" /*  load y_buf to v0    */
    "1:                             \n"
    "movi v1.4s, #0                 \n"
    "cbz  %w[with_bias], 2f         \n" /* check bias */
    "ldr  q1, [%[bias]], #16        \n" /* load bias */
    "2:                             \n"
    "fmla v1.4s, v2.4s, %[beta].4s  \n" /* y = y * beta + bias */
    "ldr  q2,  [%[out_y], #16]      \n" /*  load y to v2    */
    "fmla v1.4s, v0.4s, %[alpha].4s \n" /* out * beta */
    "ld1  {v0.4s},  [%[in_y]], #16  \n" /*   load y to v0   */
    "cbz  %w[relu], 3f              \n" /* check relu */
    "fmax v1.4s, v1.4s, %[vzero].4s \n" /* v0 relu */
    "3:                             \n"
    "subs %w[cnt],  %w[cnt], #1     \n" /*      sub cnt     */
    "st1  {v1.4s},  [%[out_y]], #16 \n" /*  store v1 to y   */
    "bne  1b                        \n" /* branch to label 1*/
    "sub  %[in_y],  %[in_y],  #16   \n" /*   restore in_y   */
    : [cnt] "+r"(cnt4), [in_y] "+r"(y_buf_ptr),
    [out_y] "+r"(y), [bias]"+r"(bias)
    : [vzero] "w"(vzero), [alpha] "w"(valpha), [beta] "w"(vbeta),
    [relu] "r" (with_relu), [with_bias] "r" (with_bias)
    : "v0", "v1", "v2", "cc", "memory");
  }

  for (int r = 0; r < remain; ++r) {
    y[r] = y_buf_ptr[r] * alpha + beta * y[r];
    if (with_bias) {
      y[r] += bias[r];
    }
    if (with_relu) {
      y[r] = y[r] > 0.f ? y[r] : 0.f;
    }
  }
}

//! define compute kernel
#ifdef __aarch64__
#else  // __aarch64__

#define SGEMV_IN_4                                                    \
  "pld [%[in]]                    @ preload cache line, input\n"      \
  "pld [%[w0]]                    @ preload cache line, weights r0\n" \
  "pld [%[w1]]                    @ preload cache line, weights r1\n" \
  "pld [%[w2]]                    @ preload cache line, weights r2\n" \
  "pld [%[w3]]                    @ preload cache line, weights r3\n" \
  "vmov.u32 q0, #0                @ set q0 to 0\n"                    \
  "vmov.u32 q1, #0                @ set q1 to 0\n"                    \
  "vmov.u32 q2, #0                @ set q2 to 0\n"                    \
  "vmov.u32 q3, #0                @ set q3 to 0\n"                    \
  "pld [%[w0], #64]               @ preload cache line, weights r0\n" \
  "pld [%[w1], #64]               @ preload cache line, weights r1\n" \
  "pld [%[w2], #64]               @ preload cache line, weights r2\n" \
  "pld [%[w3], #64]               @ preload cache line, weights r3\n"

#define SGEMV_IN_4_BIAS                                               \
  "pld [%[in]]                    @ preload cache line, input\n"      \
  "pld [%[w0]]                    @ preload cache line, weights r0\n" \
  "pld [%[w1]]                    @ preload cache line, weights r1\n" \
  "pld [%[w2]]                    @ preload cache line, weights r2\n" \
  "pld [%[w3]]                    @ preload cache line, weights r3\n" \
  "vmov.u32 q0, #0                @ set q0 to 0\n"                    \
  "vmov.u32 q1, #0                @ set q1 to 0\n"                    \
  "vmov.u32 q2, #0                @ set q2 to 0\n"                    \
  "vmov.u32 q3, #0                @ set q3 to 0\n"                    \
  "vmov s0, %[bias0]              @ set q0 to bias0\n"                \
  "vmov s4, %[bias1]              @ set q1 to bias1\n"                \
  "vmov s8, %[bias2]              @ set q2 to bias2\n"                \
  "vmov s12,%[bias3]              @ set q3 to bias3\n"                \
  "pld [%[w0], #64]               @ preload cache line, weights r0\n" \
  "pld [%[w1], #64]               @ preload cache line, weights r1\n" \
  "pld [%[w2], #64]               @ preload cache line, weights r2\n" \
  "pld [%[w3], #64]               @ preload cache line, weights r3\n"

#define SGEMV_IN_1                                                        \
  "pld [%[in]]                        @ preload cache line, input\n"      \
  "pld [%[w0]]                        @ preload cache line, weights r0\n" \
  "vmov.u32 q0, #0                    @ set q0 to 0\n"

#define SGEMV_IN_1_BIAS                                                   \
  "pld [%[in]]                        @ preload cache line, input\n"      \
  "pld [%[w0]]                        @ preload cache line, weights r0\n" \
  "vmov.u32 q0, #0                    @ set q0 to 0\n"                    \
  "vmov s0, %[bias0]                  @ set q0 to 0\n"

#define SGEMV_KERNEL_4                                                         \
  /* check main loop */                                                        \
  "cmp %[cnt], #1                 @ check whether has main loop\n"             \
  "blt  2f                        @ jump to tail\n"                            \
  "1:                             @ main loop\n"                               \
  "vld1.32 {d8-d11}, [%[in]]!     @ load input, q4, q5\n"                      \
  "vld1.32 {d12-d15}, [%[w0]]!    @ load weights r0, q6,q7\n"                  \
  "vld1.32 {d16-d19}, [%[w1]]!    @ load weights r1, q8,q9\n"                  \
  "vld1.32 {d20-d23}, [%[w2]]!    @ load weights r2, q10,q11\n"                \
  "vld1.32 {d24-d27}, [%[w3]]!    @ load weights r3, q12,q13\n"                \
  "vmla.f32 q0, q4, q6            @ mul add\n"                                 \
  "vmla.f32 q1, q4, q8            @ mul add\n"                                 \
  "vmla.f32 q2, q4, q10           @ mul add\n"                                 \
  "vmla.f32 q3, q4, q12           @ mul add\n"                                 \
  "subs %[cnt], #1                @ sub loop count \n"                         \
  "vmla.f32 q0, q5, q7            @ mul add\n"                                 \
  "vmla.f32 q1, q5, q9            @ mul add\n"                                 \
  "vmla.f32 q2, q5, q11           @ mul add\n"                                 \
  "vmla.f32 q3, q5, q13           @ mul add\n"                                 \
  "bne 1b                         @ jump to main loop\n"                       \
  /* pair add to final result */                                               \
  "2:                             @ pair add \n"                               \
  "vpadd.f32 d8, d0, d1           @ pair add, first step\n"                    \
  "vpadd.f32 d9, d2, d3           @ pair add, first step\n"                    \
  "vpadd.f32 d10, d4, d5          @ pair add, first step\n"                    \
  "vpadd.f32 d11, d6, d7          @ pair add, first step\n"                    \
  "vpadd.f32 d0, d8, d9           @ pair add, second step\n"                   \
  "vpadd.f32 d1, d10, d11         @ pair add, second step\n" /* check tails */ \
  "cmp %[tail], #1                @ check whether has tail\n"                  \
  "blt  4f                        @ jump to end\n"                             \
  "3:                             @ tail loop\n"                               \
  "vldm     %[in]!, {s16}         @ load 1 float\n"                            \
  "vldm     %[w0]!, {s17}         @ load 1 float\n"                            \
  "vldm     %[w1]!, {s18}         @ load 1 float\n"                            \
  "vldm     %[w2]!, {s19}         @ load 1 float\n"                            \
  "vldm     %[w3]!, {s20}         @ load 1 float\n"                            \
  "vmla.f32   s0, s16, s17        @ mul + add\n"                               \
  "vmla.f32   s1, s16, s18        @ mul + add\n"                               \
  "vmla.f32   s2, s16, s19        @ mul + add\n"                               \
  "vmla.f32   s3, s16, s20        @ mul + add\n"                               \
  "subs %[tail], #1               @ sub loop count \n"                         \
  "bne 3b                         @ jump to tail loop\n"

#define SGEMV_KERNEL_1                                                         \
  "cmp %[cnt], #1                     @ check whether has main loop\n"         \
  "blt  2f                            @ jump to tail\n"                        \
  "1:                                 @ main loop\n"                           \
  "vld1.32 {d24-d27}, [%[in]]!        @ load input, q12,q13\n"                 \
  "vld1.32 {d28-d31}, [%[w0]]!        @ load weights r0, q14, q15\n"           \
  "vmla.f32 q0, q12, q14              @ mul add\n"                             \
  "vmla.f32 q0, q13, q15              @ mul add\n"                             \
  "subs %[cnt] , #1                   @ sub loop count \n"                     \
  "bne 1b                             @ jump to main loop\n"                   \
  "2:                                 @ end processing\n"                      \
  "vpadd.f32 d2, d0, d1               @ pair add, first step\n"                \
  "vpadd.f32 d0, d2, d2               @ pair add, final step\n"/*check tails*/ \
  "cmp %[tail], #1                    @ check whether has mid cols\n"          \
  "blt  4f                            @ jump to end\n"                         \
  "3:                                 @ tail loop\n"                           \
  "vldm     %[in]!, {s16}             @ load 1 float\n"                        \
  "vldm     %[w0]!, {s17}             @ load 1 float\n"                        \
  "vmla.f32   s0, s16, s17            @ mul + add\n"                           \
  "subs %[tail], #1                   @ sub loop count \n"                     \
  "bne 3b                             @ jump to tail loop\n"

#define SGEMV_OUT_4                        \
  /* end */                                \
  "4:                             @ end\n" \
  "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

#define SGEMV_OUT_4_RELU                             \
  /* end */                                          \
  "4:                             @ end\n"           \
  "vmov.i32   q1, #0              @ zero for relu\n" \
  "vmax.f32   q0, q0, q1          @ relu\n"          \
  "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

#define SGEMV_OUT_1                        \
  /* end */                                \
  "4:                             @ end\n" \
  "vst1.32 {d0[0]}, [%[out]]      @ save result\n"

#define SGEMV_OUT_1_RELU                             \
  /* end */                                          \
  "4:                             @ end\n"           \
  "vmov.i32   d1, #0              @ zero for relu\n" \
  "vmax.f32   d0, d0, d1          @ relu\n"          \
  "vst1.32 {d0[0]}, [%[out]]      @ save result\n"
#endif

}  // namespace funcs
}  // namespace arm
}  // namespace onnxruntime
