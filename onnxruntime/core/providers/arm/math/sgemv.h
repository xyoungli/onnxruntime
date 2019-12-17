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

void sgemv(bool trans,
           int M,
           int N,
           float alpha,
           const float* A,
           int lda,
           const float* x,
           int incx, // not used
           float beta,
           float* y,
           int incy, // not used
           const float* bias,
           bool with_bias,
           bool with_relu,
           const ARMExecutionProvider* ctx);

}  // namespace arm
}  // namespace onnxruntime
