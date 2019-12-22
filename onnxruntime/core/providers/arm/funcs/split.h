#pragma once

#include <vector>
#include <cstring>

namespace onnxruntime {
namespace arm {
namespace funcs {

template <typename T>
void Split(const T* din,
           const std::vector<T*>& douts,
           int inner_num,
           const std::vector<int>& outer_nums) {
  assert(douts.size() == outer_nums.size());
  int strides = 0;
  for (auto& n : outer_nums) {
    strides += n;
  }
  auto ptr = din;
  for (int i = 0; i < douts.size(); i++) {
    for (int j = 0; j < inner_num; ++j) {
      std::memcpy(douts[i] + j * outer_nums[i], ptr + j * strides, sizeof(T) * outer_nums[i]);
    }
    ptr += outer_nums[i];
  }
}

}  // namespace funcs
}  // namespace arm
}  // namespace onnxruntime
