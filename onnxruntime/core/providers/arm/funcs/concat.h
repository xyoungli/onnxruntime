#pragma once

#include <vector>
#include <stdio.h>

namespace onnxruntime {
namespace arm {
namespace funcs {

template <typename T>
inline void concat(const std::vector<const T*>& din,
                   const std::vector<int>& inner_size,
                   int outer_size,
                   T* dout) {
  size_t num = din.size();
  int inner_stride = 0;
  for (auto& d : inner_size) {
    inner_stride += d;
  }
  // computation
  for (int k = 0; k < outer_size; ++k) {
    float *dst_ptr = dout + k * inner_stride;
    int col_idx = 0;
    for (size_t i = 0; i < num; ++i) {
      int col_len = inner_size[i];
      const float *src_prt = din[i] + k * col_len;
      std::memcpy(dst_ptr + col_idx, src_prt, sizeof(T) * col_len);
      col_idx += col_len;
    }
  }
}

}  // namespace funcs
}  // namespace arm
}  // namespace onnxruntime
