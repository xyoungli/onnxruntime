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

#include <cstdarg>
#include <string>
#include <vector>
#ifdef USE_OPENMP
#include <omp.h>
#endif  // USE_OPENMP

namespace onnxruntime {
namespace arm {
typedef enum {
  kAPPLE = 0,
  kA53 = 53,
  kA55 = 55,
  kA57 = 57,
  kA72 = 72,
  kA73 = 73,
  kA75 = 75,
  kA76 = 76,
  kA77 = 77,
  kARMArch_UNKOWN = -1
} ARMArch;

#if defined(USE_ARM_LINUX) || defined(USE_ANDROID)

bool BindThreads(const std::vector<int>& cpu_ids);
bool CheckCpuOnline(const std::vector<int>& cpu_ids);

#endif

struct ARMDeviceInfo {
  uint32_t core_num;
  uint32_t mem_size;
  std::vector<int> max_freqs;
  std::vector<int> min_freqs;
  std::string dev_name;

  std::vector<int> L1_cache;
  std::vector<int> L2_cache;
  std::vector<int> L3_cache;
  std::vector<int> core_ids;
  std::vector<int> big_core_ids;
  std::vector<int> little_core_ids;
  std::vector<int> cluster_ids;
  std::vector<ARMArch> archs;
  std::vector<bool> fp32;
  std::vector<bool> fp16;
  std::vector<bool> dot;
};

class ARMDevice {
public:
  static ARMDevice& Global() {
    static auto* x = new ARMDevice;
    return *x;
  }

  static int Init() {
    static int ret = Global().Setup();
    return ret;
  }

  int Setup();

  ARMArch GetArch(unsigned int active_id) const { return info_.archs[active_id > info_.core_num ? 0 : active_id]; }
  int L1Size(unsigned int active_id) const { return info_.L1_cache[active_id > info_.core_num ? 0 : active_id]; }
  int L2Size(unsigned int active_id) const { return info_.L2_cache[active_id > info_.core_num ? 0 : active_id]; }
  int L3Size(unsigned int active_id) const { return info_.L3_cache[active_id > info_.core_num ? 0 : active_id]; }
  int LLCSize(unsigned int active_id) const {
    active_id = active_id > info_.core_num ? 0 : active_id;
    auto size = info_.L3_cache[active_id] > 0 ? info_.L3_cache[active_id] : info_.L2_cache[active_id];
    return size > 0 ? size : 512 * 1024;
  }
  bool HasDot(unsigned int active_id) const { return info_.dot[active_id > info_.core_num ? 0 : active_id]; }
  bool HasFp16(unsigned int active_id) const { return info_.fp16[active_id > info_.core_num ? 0 : active_id]; }

  const ARMDeviceInfo& Info() const { return info_; }

private:
  ARMDevice() = default;
  ARMDeviceInfo info_;
  void SetDotInfo(int argc, ...);
  void SetFP16Info(int argc, ...);
  void SetFP32Info(int argc, ...);
  void SetCacheInfo(int cache_id, int argc, ...);
  void SetArchInfo(int argc, ...);
  bool SetCPUInfoByName();
  void SetCPUInfoByProb();
};

}  // namespace arm
}  // namespace onnxruntime
