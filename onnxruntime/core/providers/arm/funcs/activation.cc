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
#include "activation.h"
#include "neon_math.h"
#include <cmath>
#include <arm_neon.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace onnxruntime {
namespace arm {
namespace funcs {

const float tanh_upper = 10.f;
const float tanh_lower = -10.f;

template <>
void ActReLU<float>(const float* din, float* dout, int size) {
#ifdef USE_OPENMP
  int threads = omp_get_num_threads();
#else
  int threads = 1;
#endif
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt = nums_per_thread >> 4;
  int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
  float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for
  for (int i = 0; i < threads; ++i) {
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    for (int num = 0; num < neon_loop_cnt; ++num) {
      float32x4_t vr0 = vld1q_f32(ptr_in_thread);
      float32x4_t vr1 = vld1q_f32(ptr_in_thread + 4);
      float32x4_t vr2 = vld1q_f32(ptr_in_thread + 8);
      float32x4_t vr3 = vld1q_f32(ptr_in_thread + 12);
      ptr_in_thread += 16;
      vr0 = vmaxq_f32(vr0, vzero);
      vr1 = vmaxq_f32(vr1, vzero);
      vr2 = vmaxq_f32(vr2, vzero);
      vr3 = vmaxq_f32(vr3, vzero);
      vst1q_f32(ptr_out_thread, vr0);
      vst1q_f32(ptr_out_thread + 4, vr1);
      vst1q_f32(ptr_out_thread + 8, vr2);
      vst1q_f32(ptr_out_thread + 12, vr3);
      ptr_out_thread += 16;
    }

    for (int j = 0; j < neon_loop_remain; ++j) {
      ptr_out_thread[0] = ptr_in_thread[0] > 0.f ? ptr_in_thread[0] : 0.f;
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  float* out_ptr_remain = dout + threads * nums_per_thread;
  const float* in_ptr_remain = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    out_ptr_remain[0] = in_ptr_remain[0] > 0.f ? in_ptr_remain[0] : 0.f;
    in_ptr_remain++;
    out_ptr_remain++;
  }
}

template <>
void ActReLUNeg<float>(const float* din, float* dout, int size,
                       float negative_slope) {
#ifdef USE_OPENMP
  int threads = omp_get_num_threads();
#else
  int threads = 1;
#endif
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt = nums_per_thread >> 4;
  int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
  float32x4_t vzero = vdupq_n_f32(0.f);
  float32x4_t valpha = vdupq_n_f32(negative_slope);
#pragma omp parallel for
  for (int i = 0; i < threads; ++i) {
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    for (int num = 0; num < neon_loop_cnt; ++num) {
      float32x4_t vr0 = vld1q_f32(ptr_in_thread);
      float32x4_t vr1 = vld1q_f32(ptr_in_thread + 4);
      float32x4_t vr2 = vld1q_f32(ptr_in_thread + 8);
      float32x4_t vr3 = vld1q_f32(ptr_in_thread + 12);
      ptr_in_thread += 16;

      uint32x4_t vm0 = vcgeq_f32(vr0, vzero);
      uint32x4_t vm1 = vcgeq_f32(vr1, vzero);
      uint32x4_t vm2 = vcgeq_f32(vr2, vzero);
      uint32x4_t vm3 = vcgeq_f32(vr3, vzero);

      float32x4_t vn0 = vmulq_f32(vr0, valpha);
      float32x4_t vn1 = vmulq_f32(vr1, valpha);
      float32x4_t vn2 = vmulq_f32(vr2, valpha);
      float32x4_t vn3 = vmulq_f32(vr3, valpha);

      float32x4_t vo0 = vbslq_f32(vm0, vr0, vn0);
      float32x4_t vo1 = vbslq_f32(vm1, vr1, vn1);
      float32x4_t vo2 = vbslq_f32(vm2, vr2, vn2);
      float32x4_t vo3 = vbslq_f32(vm3, vr3, vn3);

      vst1q_f32(ptr_out_thread, vo0);
      vst1q_f32(ptr_out_thread + 4, vo1);
      vst1q_f32(ptr_out_thread + 8, vo2);
      vst1q_f32(ptr_out_thread + 12, vo3);
      ptr_out_thread += 16;
    }

    for (int j = 0; j < neon_loop_remain; ++j) {
      ptr_out_thread[0] = ptr_in_thread[0] > 0.f
                              ? ptr_in_thread[0]
                              : ptr_in_thread[0] * negative_slope;
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  float* out_ptr_remain = dout + threads * nums_per_thread;
  const float* in_ptr_remain = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    out_ptr_remain[0] = in_ptr_remain[0] > 0.f
                            ? in_ptr_remain[0]
                            : in_ptr_remain[0] * negative_slope;
    in_ptr_remain++;
    out_ptr_remain++;
  }
}

template <>
void ActClippedReLU<float>(
    const float* din, float* dout, int size, float coef) {
#ifdef USE_OPENMP
  int threads = omp_get_num_threads();
#else
  int threads = 1;
#endif
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt = nums_per_thread >> 4;
  int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
  float32x4_t vzero = vdupq_n_f32(0.f);
  float32x4_t vclip = vdupq_n_f32(coef);
#pragma omp parallel for
  for (int i = 0; i < threads; ++i) {
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    for (int num = 0; num < neon_loop_cnt; ++num) {
      float32x4_t vr0 = vld1q_f32(ptr_in_thread);
      float32x4_t vr1 = vld1q_f32(ptr_in_thread + 4);
      float32x4_t vr2 = vld1q_f32(ptr_in_thread + 8);
      float32x4_t vr3 = vld1q_f32(ptr_in_thread + 12);
      ptr_in_thread += 16;
      float32x4_t vt0 = vmaxq_f32(vr0, vzero);
      float32x4_t vt1 = vmaxq_f32(vr1, vzero);
      float32x4_t vt2 = vmaxq_f32(vr2, vzero);
      float32x4_t vt3 = vmaxq_f32(vr3, vzero);

      float32x4_t vo0 = vminq_f32(vt0, vclip);
      float32x4_t vo1 = vminq_f32(vt1, vclip);
      float32x4_t vo2 = vminq_f32(vt2, vclip);
      float32x4_t vo3 = vminq_f32(vt3, vclip);

      vst1q_f32(ptr_out_thread, vo0);
      vst1q_f32(ptr_out_thread + 4, vo1);
      vst1q_f32(ptr_out_thread + 8, vo2);
      vst1q_f32(ptr_out_thread + 12, vo3);
      ptr_out_thread += 16;
    }
    for (int j = 0; j < neon_loop_remain; ++j) {
      ptr_out_thread[0] = ptr_in_thread[0] > 0.f ? ptr_in_thread[0] : 0.f;
      ptr_out_thread[0] = ptr_out_thread[0] < coef ? ptr_out_thread[0] : coef;
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  float* out_ptr_remain = dout + threads * nums_per_thread;
  const float* in_ptr_remain = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    out_ptr_remain[0] = in_ptr_remain[0] > 0.f ? in_ptr_remain[0] : 0.f;
    out_ptr_remain[0] = out_ptr_remain[0] < coef ? out_ptr_remain[0] : coef;
    in_ptr_remain++;
    out_ptr_remain++;
  }
}

template <>
void ActPreLU<float>(const float* din, float* dout,
                     int outer_size, int channel_size, int inner_size,
                     const std::string& mode, const float* alpha_data) {
  if (mode == "all" || mode == "channel") {
    int stride_size = inner_size * channel_size;
    int cnt = inner_size >> 4;
    int remain = inner_size & 15;
    float32x4_t vzero = vdupq_n_f32(0.f);
    for (int n = 0; n < outer_size; n++) {
      const float* data_in_batch = din + n * stride_size;
      float* data_out_batch = dout + n * stride_size;
#pragma omp parallel for
      for (int c = 0; c < channel_size; c++) {
        const float* data_in_c = data_in_batch + c * inner_size;
        float* data_out_c = data_out_batch + c * inner_size;

        float slope = mode == "all" ? alpha_data[0] : alpha_data[c];
        float32x4_t vslope = vdupq_n_f32(slope);
#ifdef __aarch64__
        for (int i = 0; i < cnt; ++i) {
          float32x4_t vr0 = vld1q_f32(data_in_c);
          float32x4_t vr1 = vld1q_f32(data_in_c + 4);
          float32x4_t vr2 = vld1q_f32(data_in_c + 8);
          float32x4_t vr3 = vld1q_f32(data_in_c + 12);
          uint32x4_t vm0 = vcltq_f32(vr0, vzero);    // vr0 <= vzero
          uint32x4_t vm1 = vcltq_f32(vr1, vzero);    // vr0 <= vzero
          uint32x4_t vm2 = vcltq_f32(vr2, vzero);    // vr0 <= vzero
          uint32x4_t vm3 = vcltq_f32(vr3, vzero);    // vr0 <= vzero
          float32x4_t vo0 = vmulq_f32(vr0, vslope);  // vr0 * vslope
          float32x4_t vo1 = vmulq_f32(vr1, vslope);  // vr0 * vslope
          float32x4_t vo2 = vmulq_f32(vr2, vslope);  // vr0 * vslope
          float32x4_t vo3 = vmulq_f32(vr3, vslope);  // vr0 * vslope
          float32x4_t vos0 = vbslq_f32(vm0, vo0, vr0);
          float32x4_t vos1 = vbslq_f32(vm1, vo1, vr1);
          float32x4_t vos2 = vbslq_f32(vm2, vo2, vr2);
          float32x4_t vos3 = vbslq_f32(vm3, vo3, vr3);
          vst1q_f32(data_out_c, vos0);
          vst1q_f32(data_out_c + 4, vos1);
          vst1q_f32(data_out_c + 8, vos2);
          vst1q_f32(data_out_c + 12, vos3);
          data_in_c += 16;
          data_out_c += 16;
        }
#else
        int cnt_loop = cnt;
        if (cnt_loop > 0) {
          asm volatile(
              "vld1.32    {d0-d3}, [%[ptr_in]]!                       @ load "
              "input to q0, q1\n"
              "pld [%[ptr_in]]                                @ preload\n"
              "pld [%[ptr_in], #64]                           @ preload\n"
              "pld [%[ptr_in], #128]                          @ preload\n"
              "pld [%[ptr_in], #192]                          @ preload\n"
              "1:                                             @main loop\n"
              "vld1.32    {d4-d7}, [%[ptr_in]]!               @ load input to "
              "q2, q3\n"
              "vclt.f32   q8, q0, %q[vzero]                   @vcle q0 <= "
              "vzero\n"
              "vclt.f32   q9, q1, %q[vzero]                   @vcle q1 <= "
              "vzero\n"
              "vmul.f32  q10, q0, %q[vslope]                  @vmul q0 * "
              "vslope\n"
              "vmul.f32  q11, q1, %q[vslope]                  @vmul q1 * "
              "vslope\n"

              "vclt.f32  q12, q2, %q[vzero]                   @vcle q2 <= "
              "vzero\n"
              "vclt.f32  q13, q3, %q[vzero]                   @vcle q3 <= "
              "vzero\n"
              "vmul.f32  q14, q2, %q[vslope]                  @vmul q2 * "
              "vslope\n"
              "vmul.f32  q15, q3, %q[vslope]                  @vmul q3 * "
              "vslope\n"

              "vbif.32    q10, q0, q8                         @vbit q10, q0, "
              "q8\n"
              "vbif.32    q11, q1, q9                         @vbit q11, q1, "
              "q9\n"
              "vbif.32    q14, q2, q12                        @vbit q14, q2, "
              "q12\n"
              "vbif.32    q15, q3, q13                        @vbit q15, q3, "
              "q13\n"

              "subs       %[cnt], #1                          @subs nn, 1\n"
              "vld1.32    {d0-d3}, [%[ptr_in]]!               @ load input to "
              "q0, q1\n"

              "vst1.f32   {d20-d23}, [%[dout]]!               @store data\n"
              "vst1.f32   {d28-d31}, [%[dout]]!               @store data\n"
              "bne        1b                                  @bne nn\n"
              "sub    %[ptr_in], #32                          @ ptr-32\n"
              : [ptr_in] "+r"(data_in_c),
                [cnt] "+r"(cnt_loop),
                [dout] "+r"(data_out_c)
              : [vzero] "w"(vzero), [vslope] "w"(vslope)
              : "cc",
                "memory",
                "q0",
                "q1",
                "q2",
                "q3",
                "q8",
                "q9",
                "q10",
                "q11",
                "q12",
                "q13",
                "q14",
                "q15");
        }
#endif  // __aarch64__
        for (int i = remain; i > 0; i--) {
          *(data_out_c++) =
              data_in_c[0] > 0.f ? data_in_c[0] : data_in_c[0] * slope;
          data_in_c++;
        }
      }
    }
  } else {  // mode = element
    int stride_size = inner_size * channel_size;
    for (int n = 0; n < outer_size; n++) {
      const float* data_in_batch = din + n * stride_size;
      const float* data_alpha_batch = alpha_data + n * stride_size;
      float* data_out_batch = dout + n * stride_size;
      for (int c = 0; c < channel_size; c++) {
        const float* data_in_c = data_in_batch + c * inner_size;
        const float* data_alpha_c = data_alpha_batch + c * inner_size;
        float* data_out_c = data_out_batch + c * inner_size;
        for (int i = 0; i < inner_size; i++) {
          data_out_c[0] = data_in_c[0] > 0.f ? data_in_c[0]
                                             : data_in_c[0] * data_alpha_c[0];
          data_in_c++;
          data_alpha_c++;
          data_out_c++;
        }
      }
    }
  }
}

template <>
void ActSigmoid<float>(const float* din, float* dout, int size) {
#ifdef USE_OPENMP
  int threads = omp_get_num_threads();
#else
  int threads = 1;
#endif
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt_dim4 = nums_per_thread >> 2;
  int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);
#pragma omp parallel for
  for (int i = 0; i < threads; ++i) {
    float32x4_t exp_vec;
    float32x4_t recip;
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    for (int k = 0; k < neon_loop_cnt_dim4; ++k) {
      exp_vec = vexpq_f32(vnegq_f32(vld1q_f32(ptr_in_thread)));
      exp_vec = vaddq_f32(exp_vec, vdupq_n_f32(1.0f));
      recip = vrecpeq_f32(exp_vec);
      recip = vmulq_f32(vrecpsq_f32(exp_vec, recip), recip);
      recip = vmulq_f32(vrecpsq_f32(exp_vec, recip), recip);
      vst1q_f32(ptr_out_thread, recip);
      ptr_out_thread += 4;
      ptr_in_thread += 4;
    }
    for (int j = 0; j < neon_loop_remain_dim4; ++j) {
      ptr_out_thread[0] = 1.f / (1 + expf(-ptr_in_thread[0]));
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  float* ptr_out = dout + threads * nums_per_thread;
  const float* ptr_in = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    ptr_out[0] = 1.f / (1 + expf(-ptr_in[0]));
    ptr_in++;
    ptr_out++;
  }
}

// tanh : (exp(x) - exp(-x)) / (exp(x) + exp(-x))
template <>
void ActTanh<float>(const float* din, float* dout, int size) {
#ifdef USE_OPENMP
  int threads = omp_get_num_threads();
#else
  int threads = 1;
#endif
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt_dim4 = nums_per_thread >> 2;
  int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);
#pragma omp parallel for
  for (int i = 0; i < threads; ++i) {
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    for (int k = 0; k < neon_loop_cnt_dim4; ++k) {
      vst1q_f32(ptr_out_thread, vtanhq_f32(vld1q_f32(ptr_in_thread)));
      ptr_out_thread += 4;
      ptr_in_thread += 4;
    }
    for (int j = 0; j < neon_loop_remain_dim4; ++j) {
      if (*ptr_in_thread >= tanh_upper) {
        *ptr_out_thread = 1.f;
      } else if (*ptr_in_thread <= tanh_lower) {
        *ptr_out_thread = -1.f;
      } else {
        ptr_out_thread[0] = (expf(ptr_in_thread[0]) - expf(-ptr_in_thread[0])) /
                     (expf(ptr_in_thread[0]) + expf(-ptr_in_thread[0]));
      }
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  float* ptr_out = dout + threads * nums_per_thread;
  const float* ptr_in = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    if (*ptr_in >= tanh_upper) {
      *ptr_out = 1.f;
    } else if (*ptr_in <= tanh_lower) {
      *ptr_out = -1.f;
    } else {
      ptr_out[0] = (expf(ptr_in[0]) - expf(-ptr_in[0])) /
                   (expf(ptr_in[0]) + expf(-ptr_in[0]));
    }
    ptr_in++;
    ptr_out++;
  }
}

// swish: x /(1 + exp(-(b * x)))
template <>
void ActSwish<float>(const float* din, float* dout, int size, float coef) {
#ifdef USE_OPENMP
  int threads = omp_get_num_threads();
#else
  int threads = 1;
#endif
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt_dim4 = nums_per_thread >> 2;
  int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);
  const float beta = coef;
  float32x4_t vbeta = vdupq_n_f32(beta);
  float32x4_t vone = vdupq_n_f32(1.f);
#pragma omp parallel for
  for (int i = 0; i < threads; ++i) {
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    for (int k = 0; k < neon_loop_cnt_dim4; ++k) {
      float32x4_t va = vld1q_f32(ptr_in_thread);             // x
      float32x4_t vb = vnegq_f32(vld1q_f32(ptr_in_thread));  // -x
      float32x4_t vsum = vmulq_f32(vb, vbeta);
      vsum = vexpq_f32(vsum);
      float32x4_t vc = vaddq_f32(vone, vsum);
      float32x4_t vrst = vdivq_f32(va, vc);
      vst1q_f32(ptr_out_thread, vrst);
      ptr_out_thread += 4;
      ptr_in_thread += 4;
    }
    for (int j = 0; j < neon_loop_remain_dim4; ++j) {
      ptr_out_thread[0] =
          ptr_in_thread[0] / (1.0 + expf(-ptr_in_thread[0] * beta));
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  float* ptr_out = dout + threads * nums_per_thread;
  const float* ptr_in = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    ptr_out[0] = ptr_in[0] / (1.0 + expf(-ptr_in[0] * beta));
    ptr_in++;
    ptr_out++;
  }
}

template <>
void ActLog<float>(const float* din, float* dout, int size) {
#ifdef USE_OPENMP
  int threads = omp_get_num_threads();
#else
  int threads = 1;
#endif
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt_dim4 = nums_per_thread >> 2;
  int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);
#pragma omp parallel for
  for (int i = 0; i < threads; ++i) {
    float32x4_t exp_vec = vdupq_n_f32(0.0f);
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    for (int k = 0; k < neon_loop_cnt_dim4; ++k) {
      exp_vec = vlogq_f32(vld1q_f32(ptr_in_thread));
      vst1q_f32(ptr_out_thread, exp_vec);
      ptr_out_thread += 4;
      ptr_in_thread += 4;
    }
    for (int j = 0; j < neon_loop_remain_dim4; ++j) {
      ptr_out_thread[0] = logf(ptr_in_thread[0]);
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  float* ptr_out = dout + threads * nums_per_thread;
  const float* ptr_in = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    ptr_out[0] = logf(ptr_in[0]);
    ptr_in++;
    ptr_out++;
  }
}

template <>
void ActExp<float>(const float* din, float* dout, int size) {
#ifdef USE_OPENMP
  int threads = omp_get_num_threads();
#else
  int threads = 1;
#endif
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt_dim4 = nums_per_thread >> 2;
  int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);

#pragma omp parallel for
  for (int i = 0; i < threads; ++i) {
    float32x4_t exp_vec = vdupq_n_f32(0.0f);
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    for (int k = 0; k < neon_loop_cnt_dim4; ++k) {
      exp_vec = vexpq_f32(vld1q_f32(ptr_in_thread));
      vst1q_f32(ptr_out_thread, exp_vec);
      ptr_out_thread += 4;
      ptr_in_thread += 4;
    }
    for (int j = 0; j < neon_loop_remain_dim4; ++j) {
      ptr_out_thread[0] = expf(ptr_in_thread[0]);
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  float* ptr_out = dout + threads * nums_per_thread;
  const float* ptr_in = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    ptr_out[0] = expf(ptr_in[0]);
    ptr_in++;
    ptr_out++;
  }
}

template <>
void ActFloor<float>(const float* din, float* dout, int size) {
#ifdef USE_OPENMP
  int threads = omp_get_num_threads();
#else
  int threads = 1;
#endif
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt = nums_per_thread >> 4;
  int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
#pragma omp parallel for
  for (int i = 0; i < threads; ++i) {
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    for (int num = 0; num < neon_loop_cnt; ++num) {
      float32x4_t vr0 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      float32x4_t vr1 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      float32x4_t vr2 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      float32x4_t vr3 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      vr0 = vfloorq_f32(vr0);
      vr1 = vfloorq_f32(vr1);
      vr2 = vfloorq_f32(vr2);
      vr3 = vfloorq_f32(vr3);
      vst1q_f32(ptr_out_thread, vr0);
      ptr_out_thread += 4;
      vst1q_f32(ptr_out_thread, vr1);
      ptr_out_thread += 4;
      vst1q_f32(ptr_out_thread, vr2);
      ptr_out_thread += 4;
      vst1q_f32(ptr_out_thread, vr3);
      ptr_out_thread += 4;
    }
    for (int j = 0; j < neon_loop_remain; ++j) {
      ptr_out_thread[0] = floorf(ptr_in_thread[0]);
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  float* out_ptr_remain = dout + threads * nums_per_thread;
  const float* in_ptr_remain = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    out_ptr_remain[0] = floorf(in_ptr_remain[0]);
    in_ptr_remain++;
    out_ptr_remain++;
  }
}

//template <>
//void ActHardSigmoid<float>(const float* din, float* dout,
//                           int64_t size, float slope, float offset) {
//  (void)(threads);
//  for (int64_t i = 0; i < size; ++i) {
//    dout[0] = din[0] * slope + offset;
//    dout[0] = dout[0] < 1.0f ? dout[0] : 1.0f;
//    dout[0] = dout[0] > 0.0f ? dout[0] : 0.0f;
//    ++din;
//    ++dout;
//  }
//}
//
//template <>
//void ActRsqrt<float>(const float* din, float* dout, int size) {
//  const float* ptr_in = din;
//  float* ptr_out = dout;
//  for (int i = 0; i < size; ++i) {
//    ptr_out[0] = 1.0 / sqrtf(ptr_in[0]);
//    ptr_in++;
//    ptr_out++;
//  }
//}

}  // namespace funcs
}  // namespace arm
}  // namespace onnxruntime