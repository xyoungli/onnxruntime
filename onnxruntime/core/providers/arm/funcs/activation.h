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
#include <string>

namespace onnxruntime {
namespace arm {
namespace funcs {

template <typename T>
void ActReLU(const T* din, T* dout, int size);

template <typename T>
void ActReLUNeg(
    const T* din, T* dout, int size, float negative_slope);

template <typename T>
void ActClippedReLU(const T* din, T* dout, int size, float coef);

template <typename T>
void ActPreLU(const T* din, T* dout,
              int outer_size, int channel_size, int inner_size,
              const std::string& mode, const float* alpha_data);

template <typename T>
void ActSigmoid(const T* din, T* dout, int size);

template <typename T>
void ActTanh(const T* din, T* dout, int size);

template <typename T>
void ActSwish(const T* din, T* dout, int size, float coef);

template <typename T>
void ActLog(const T* din, T* dout, int size);

template <typename T>
void ActExp(const T* din, T* dout, int size);

template <typename T>
void ActFloor(const T* din, T* dout, int size);

//template <typename T>
//void ActHardSigmoid(const T* din, T* dout,
//                    int64_t size, float slope, float offset);
//
//template <typename T>
//void ActRsqrt(const T* din, T* dout, int size);

}  // namespace funcs
}  // namespace arm
}  // namespace onnxruntime
