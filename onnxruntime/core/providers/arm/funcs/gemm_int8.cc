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

#include "gemm_int8.h"

namespace onnxruntime {
namespace arm {
namespace funcs {

template <typename Dtype>
void GemmInt8(bool is_transA,
              bool is_transB,
              int M,
              int N,
              int K,
              const int8_t* A,
              const int8_t* B,
              Dtype* C,
              const float* bias,
              bool is_bias,
              bool is_relu,
              const float* scale,
              ARMExecutionProvider* ctx) {
  auto alloc_ptr = ctx->GetAllocator(0, OrtMemTypeDefault);
  int hblock = GetHblockInt8(ctx);
  int m_roundup = hblock * ((M + hblock - 1) / hblock);
  auto packed_A = static_cast<int8_t*>(
      alloc_ptr->Alloc(m_roundup * K * sizeof(int8_t)));

  int lda = is_transA ? M : K;
  PrepackAInt8(packed_A, A, lda, 0, M, 0, K, is_transA, ctx);

  GemmPrepackInt8(
      packed_A, B, bias, C, M, N, K, is_bias, is_relu, is_transB, scale, ctx);
  alloc_ptr->Free(packed_A);
}

template void GemmInt8<float>(bool is_transA,
                              bool is_transB,
                              int M,
                              int N,
                              int K,
                              const int8_t* A,
                              const int8_t* B,
                              float* C,
                              const float* bias,
                              bool is_bias,
                              bool is_relu,
                              const float* scale,
                              ARMExecutionProvider* ctx);

template void GemmInt8<int8_t>(bool is_transA,
                               bool is_transB,
                               int M,
                               int N,
                               int K,
                               const int8_t* A,
                               const int8_t* B,
                               int8_t* C,
                               const float* bias,
                               bool is_bias,
                               bool is_relu,
                               const float* scale,
                               ARMExecutionProvider* ctx);

}  // namespace funcs
}  // namespace arm
}  // namespace onnxruntime
