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

#include <random>
template <typename Dtype>
inline void fill_data_const(Dtype* dio, Dtype value, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    dio[i] = value;
  }
}

template <typename Dtype>
inline void fill_data_rand(Dtype* dio, Dtype vstart, Dtype vend, size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 1.f);
  for (size_t i = 0; i < size; ++i) {
    dio[i] = static_cast<Dtype>(vstart + (vend - vstart) * dis(gen));
  }
}

template <typename Dtype>
inline void check_precision(const Dtype* in1, const Dtype* in2, double& max_ratio, double& max_diff, size_t size) {
  const double eps = 1e-6f;
  max_diff = fabs(in1[0] - in2[0]);
  max_ratio = fabs(max_diff) / (std::abs(in1[0]) + eps);
  for (size_t i = 1; i < size; ++i) {
    double diff = fabs(in1[i] - in2[i]);
    double ratio = fabs(diff) / (std::abs(in1[i]) + eps);
    if (max_ratio < ratio) {
      max_diff = diff;
      max_ratio = ratio;
    }
  }
}

template <typename Dtype>
inline void compute_data_diff(const Dtype* in1, const Dtype* in2, Dtype* out, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    out[i] = in1[i] - in2[i];
  }
}

template <typename Dtype>
inline void print_data(const Dtype* in, size_t size, size_t stride);

template <>
inline void print_data(const float* in, size_t size, size_t stride) {
  for (size_t i = 0; i < size; ++i) {
    printf("%0.6f ", in[i]);
    if ((i + 1) % stride == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

template <>
inline void print_data(const int* in, size_t size, size_t stride) {
  for (size_t i = 0; i < size; ++i) {
    printf("%d ", in[i]);
    if ((i + 1) % stride == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

template <>
inline void print_data(const int8_t* in, size_t size, size_t stride) {
  for (size_t i = 0; i < size; ++i) {
    printf("%d ", in[i]);
    if ((i + 1) % stride == 0) {
      printf("\n");
    }
  }
  printf("\n");
}