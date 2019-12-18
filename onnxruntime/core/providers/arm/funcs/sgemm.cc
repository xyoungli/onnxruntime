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

#include "sgemm.h"
#include "sgemm_packed.h"

namespace onnxruntime {
namespace arm {

void Sgemm(bool transA,
           bool transB,
           int M,
           int N,
           int K,
           float alpha,
           const float* A,
           int lda,
           const float* B,
           int ldb,
           float beta,
           float* C,
           int ldc,
           const float* bias,
           bool is_bias,
           bool is_relu,
           ARMExecutionProvider* ctx) {
  int hblock = GetSgemmHblock(ctx);
  int m_roundup = hblock * ((M + hblock - 1) / hblock);
  auto alloc_ptr = ctx->GetAllocator(0, OrtMemTypeDefault);

  auto packed_A_ptr = static_cast<float*>(
          alloc_ptr->Alloc(m_roundup * K * sizeof(float)));

  PrepackA(packed_A_ptr, A, alpha, lda, 0, M, 0, K, transA, ctx);

  SgemmPrepack(transB, M, N, K,
               packed_A_ptr, B, ldb,
               beta, C, ldc,
               bias, is_bias, is_relu, ctx);
  alloc_ptr->Free(packed_A_ptr);
}

}  // namespace arm
}  // namespace onnxruntime
