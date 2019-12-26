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

#include <vector>

namespace onnxruntime {
namespace arm {
namespace funcs {

void Fp32ToInt32(const float* din,
                 int* dout,
                 const float* scale,
                 int axis_size,
                 int64_t outer_size,
                 int64_t inner_size);

void Int32ToFp32(const int* din,
                 float* dout,
                 const float* scale,
                 int axis_size,
                 int64_t outer_size,
                 int64_t inner_size);

void Fp32ToInt16(const float* din,
                 int16_t* dout,
                 const float* scale,
                 int axis_size,
                 int64_t outer_size,
                 int64_t inner_size);

void Int16ToFp32(const int16_t* in,
                 float* out,
                 const float* scale,
                 int axis_size,
                 int64_t outer_size,
                 int64_t inner_size);

void Fp32ToInt8(const float* din,
                int8_t* dout,
                const float* scale,
                int axis_size,
                int64_t outer_size,
                int64_t inner_size);

void Int8ToFp32(const int8_t* in,
                float* out,
                const float* scale,
                int axis_size,
                int64_t outer_size,
                int64_t inner_size);

void Int32ToInt8(const int* din,
                 int8_t* dout,
                 const float* scale,
                 int axis_size,
                 int64_t outer_size,
                 int64_t inner_size);

}  // namespace funcs
}  // namespace arm
}  // namespace onnxruntime
