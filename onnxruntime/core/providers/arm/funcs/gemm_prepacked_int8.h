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

#pragma once
#include <cmath>
#include "core/providers/arm/arm_execution_provider.h"

namespace onnxruntime {
namespace arm {
namespace funcs {

const int KBLOCK_INT8 = 4;
#ifdef __aarch64__
const int MBLOCK_INT8_OTH = 4;
const int NBLOCK_INT8_OTH = 16;

const int MBLOCK_INT8_DOT = 8;
const int NBLOCK_INT8_DOT = 12;

inline int GetHblockInt8(ARMExecutionProvider* ctx) {
  (void)(ctx);
#ifdef WITH_ARM_DOTPROD
  if (ctx->has_dot()) {
    return MBLOCK_INT8_DOT;
  } else {
    return MBLOCK_INT8_OTH;
  }
#else
  return MBLOCK_INT8_OTH;
#endif
}
#else
// const int HBLOCK = 4;
// const int WBLOCK = 8;
const int MBLOCK_INT8_OTH = 4;
const int NBLOCK_INT8_OTH = 8;

inline int GetHblockInt8(ARMExecutionProvider* ctx) { return 4; }
#endif  // __aarch64__

void PrepackAInt8(void* out,
                  const void* in,
                  int ldin,
                  int m0,
                  int mmax,
                  int k0,
                  int kmax,
                  bool is_trans,
                  ARMExecutionProvider* ctx);

template <typename dtype>
void GemmPrepackInt8(const int8_t* A_packed,
                     const int8_t* B,
                     const float* bias,
                     dtype* C,
                     int M,
                     int N,
                     int K,
                     bool is_bias,
                     bool is_relu,
                     bool is_transB,
                     const float* scale,
                     ARMExecutionProvider* ctx);

#define ROUNDUP(a, b) ((((a) + (b)-1) / (b)) * (b))

}  // namespace funcs
}  // namespace arm
}  // namespace onnxruntime
