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

//void sgemm(bool is_transA,
//           bool is_transB,
//           bool packedA,
//           bool packedB,
//           int M,
//           int N,
//           int K,
//           float alpha,
//           const float* A,
//           int lda,
//           const float* B,
//           int ldb,
//           float beta,
//           float* C,
//           int ldc,
//           const float* bias,
//           bool is_bias,
//           bool is_relu,
//           ARMExecutionProvider* ctx) {
//  int hblock = get_hblock(ctx);
//  int m_roundup = hblock * ((M + hblock - 1) / hblock);
//}

}  // namespace arm
}  // namespace onnxruntime
