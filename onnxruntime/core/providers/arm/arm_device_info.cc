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

// Parts of the following code in this file refs to
// https://github.com/Tencent/ncnn/blob/master/src/cpu.cpp
// Tencent is pleased to support the open source community by making ncnn
// available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#if defined(USE_ARM_LINUX) || defined (USE_ANDROID)
#include <sys/syscall.h>
#include <unistd.h>
#include <sched.h>
#endif  // USE_ARM_LINUX

#ifdef USE_ANDROID
#include <sys/system_properties.h>
#endif  //  USE_ANDROID

#if __APPLE__
#include "TargetConditionals.h"
#ifdef USE_IOS
#include <mach/machine.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#endif  // USE_IOS
#endif  // __APPLE__

#include <algorithm>
#include <limits>
#include "arm_device_info.h"

namespace onnxruntime {
namespace arm {

#ifdef USE_IOS
const int DEFAULT_L1_CACHE_SIZE = 64 * 1024;
const int DEFAULT_L2_CACHE_SIZE = 2048 * 1024;
const int DEFAULT_L3_CACHE_SIZE = 0;
#else
const int DEFAULT_L1_CACHE_SIZE = 32 * 1024;
const int DEFAULT_L2_CACHE_SIZE = 512 * 1024;
const int DEFAULT_L3_CACHE_SIZE = 0;
#endif

int GetCpuNum() {
#if defined(USE_ARM_LINUX) || defined(USE_ANDROID)
  // get cpu count from /sys/devices/system/cpu/cpunum/uevent
  int max_cpu_num = 20;
  int cpu_num = 0;
  for (int i = 0; i < max_cpu_num; ++i) {
    char path[256];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/uevent", i);
    FILE* fp = fopen(path, "rb");
    if (!fp) {
      break;
    }
    cpu_num++;
    fclose(fp);
  }
  if (cpu_num < 1) {
    cpu_num = 1;
  }
  return cpu_num;
#elif defined(USE_IOS)
  int cpu_num = 0;
  size_t len = sizeof(cpu_num);
  sysctlbyname("hw.ncpu", &cpu_num, &len, NULL, 0);
  if (cpu_num < 1) {
    cpu_num = 1;
  }
  return cpu_num;
#else
  return 1;
#endif
}

size_t GetMemSize() {
#if defined(USE_ARM_LINUX) || defined(USE_ANDROID)
  // get cpu count from /proc/cpuinfo
  FILE* fp = fopen("/proc/meminfo", "rb");
  if (!fp) {
    return 1;
  }
  size_t memsize = 0;
  char line[1024];
  while (!feof(fp)) {
    char* s = fgets(line, 1024, fp);
    if (!s) {
      break;
    }
    sscanf(s, "MemTotal:        %zd kB", &memsize);
  }
  fclose(fp);
  return memsize;
#elif defined(USE_IOS)
  // to be implemented
  printf("not implemented, set to default 4GB\n");
  return 4096 * 1024;
#else
  return 0;
#endif
}

void GetCpuArch(std::vector<ARMArch>& archs, const int cpu_num) {
  archs.resize(cpu_num);
  for (int i = 0; i < cpu_num; ++i) {
    archs[i] = kARMArch_UNKOWN;
  }
#if defined(USE_ARM_LINUX) || defined(USE_ANDROID)
  //! get CPU ARCH
  FILE* fp = fopen("/proc/cpuinfo", "rb");
  if (!fp) {
    return;
  }
  int cpu_idx = 0;
  char line[1024];
  while (!feof(fp)) {
    char* s = fgets(line, 1024, fp);
    if (!s) {
      break;
    }
    if (strstr(line, "part") != nullptr) {
      ARMArch arch_type = kARMArch_UNKOWN;
      int arch_id = 0;
      sscanf(s, "CPU part\t: %x", &arch_id);
      switch (arch_id) {
        case 0xd03:
          arch_type = kA53;
          break;
        case 0xd05:
          arch_type = kA55;
          break;
        case 0xd07:
          arch_type = kA57;
          break;
        case 0xd08:
          arch_type = kA72;
          break;
        case 0xd09:
          arch_type = kA73;
          break;
        case 0xd0a:
          arch_type = kA75;
          break;
        case 0xd40:
          arch_type = kA76;
          break;
        case 0x804:
          // 855
          arch_type = kA76;
          break;
        case 0x805:
          // 855
          arch_type = kA55;
          break;
        case 0x802:
          // 845
          arch_type = kA75;
          break;
        case 0x803:
          // 845
          arch_type = kA55;
          break;
        case 0x801:
          // 835
          arch_type = kA73;
          break;
        case 0x800:
          // 835
          arch_type = kA73;
          break;
        case 0x205:
          // 820
          arch_type = kA72;
          break;
        default:
          printf("Unknow cpu arch: %d\n", arch_id);
      }
      archs[cpu_idx] = arch_type;
      cpu_idx++;
    }
  }
  fclose(fp);
  for (; cpu_idx > 0 && cpu_idx < cpu_num; ++cpu_idx) {
    archs[cpu_idx] = archs[cpu_idx - 1];
  }
#elif defined(USE_IOS)
  for (int i = 0; i < cpu_num; ++i) {
    archs[i] = kAPPLE;
  }
#endif
}

#if defined(USE_ARM_LINUX) || defined(USE_ANDROID)

std::string GetCpuName() {
  std::string cpu_name;
  FILE* fp = fopen("/proc/cpuinfo", "rb");
  if (!fp) {
    return "";
  }
  char line[1024];
  while (!feof(fp)) {
    char* s = fgets(line, 1024, fp);
    if (!s) {
      break;
    }
    if (strstr(line, "Hardware") != NULL) {
      cpu_name = std::string(line);
    }
  }
#ifdef USE_ANDROID
  // cpu name concat board name, platform name and chip name
  char board_name[128];
  char platform_name[128];
  char chip_name[128];
  __system_property_get("ro.product.board", board_name);
  __system_property_get("ro.board.platform", platform_name);
  __system_property_get("ro.chipname", chip_name);
  cpu_name =
      cpu_name + "_" + board_name + "_" + platform_name + "_" + chip_name;
#endif
  std::transform(cpu_name.begin(), cpu_name.end(), cpu_name.begin(), ::toupper);
  fclose(fp);
  return cpu_name;
}

int GetMinFreqKhz(int cpuid) {
  // first try, for all possible cpu
  char path[256];
  snprintf(path,
           sizeof(path),
           "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",
           cpuid);
  FILE* fp = fopen(path, "rb");
  if (!fp) {
    return -1;
  }

  int min_freq_khz = -1;
  fscanf(fp, "%d", &min_freq_khz);
  fclose(fp);
  return min_freq_khz;
}

int GetMaxFreqKhz(int cpuid) {
  // first try, for all possible cpu
  char path[256];
  snprintf(path,
           sizeof(path),
           "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state",
           cpuid);

  FILE* fp = fopen(path, "rb");
  if (!fp) {
    // second try, for online cpu
    snprintf(path,
             sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state",
             cpuid);
    fp = fopen(path, "rb");
  }

  int max_freq_khz = 0;
  if (fp) {
    while (!feof(fp)) {
      int freq_khz = 0;
      int nscan = fscanf(fp, "%d %*d", &freq_khz);
      if (nscan != 1) {
        break;
      }

      if (freq_khz > max_freq_khz) {
        max_freq_khz = freq_khz;
      }
    }
  }
  if (max_freq_khz == 0 || !fp) {
    // third try, for online cpu
    snprintf(path,
             sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",
             cpuid);
    fp = fopen(path, "rb");
    if (!fp) {
      return -1;
    }
    int max_freq_khz = -1;
    fscanf(fp, "%d", &max_freq_khz);
    fclose(fp);
    return max_freq_khz;
  }

  fclose(fp);
  return max_freq_khz;
}

void SortCpuidByMaxFreq(const std::vector<int>& max_freqs,
        std::vector<int>* cpu_ids,
        std::vector<int>* cluster_ids) {
  int cpu_num = max_freqs.size();
  if (cpu_num == 0) {
    return;
  }
  cpu_ids->resize(cpu_num);
  cluster_ids->resize(cpu_num);
  for (int i = 0; i < cpu_num; i++) {
    cpu_ids->at(i) = i;
  }
  // sort cpuid as big core first
  // simple bubble sort
  for (int i = 0; i < cpu_num; i++) {
    for (int j = i + 1; j < cpu_num; j++) {
      if (max_freqs[i] < max_freqs[j]) {
        // swap
        int tmp = cpu_ids->at(i);
        cpu_ids->at(i) = cpu_ids->at(j);
        cpu_ids->at(j) = tmp;
      }
    }
  }
  // SMP
  int mid_max_freq =
      (max_freqs[cpu_ids->at(0)] + max_freqs[cpu_ids->at(cpu_num - 1)]) / 2;

  for (int i = 0; i < cpu_num; i++) {
    cpu_ids->at(i) = i;
    if (max_freqs[i] >= mid_max_freq) {
      cluster_ids->at(i) = 0;
    } else {
      cluster_ids->at(i) = 1;
    }
  }
}

void GetCpuCacheSize(int cpu_id,
        int& l1_cache_size,
        int& l2_cache_size,
        int& l3_cache_size) {
  int max_cache_idx_num = 10;
  l1_cache_size = DEFAULT_L1_CACHE_SIZE;
  l2_cache_size = DEFAULT_L2_CACHE_SIZE;
  l3_cache_size = DEFAULT_L3_CACHE_SIZE;
  for (int i = 0; i < max_cache_idx_num; i++) {
    char path[256];
    snprintf(path,
             sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cache/index%d/level",
             cpu_id,
             i);
    FILE* fp = fopen(path, "rb");
    if (fp) {
      int level = -1;
      fscanf(fp, "%d", &level);
      fclose(fp);
      snprintf(path,
               sizeof(path),
               "/sys/devices/system/cpu/cpu%d/cache/index%d/size",
               cpu_id,
               i);
      fp = fopen(path, "rb");
      if (fp) {
        int size = -1;
        fscanf(fp, "%d", &size);
        fclose(fp);
        if (size >= 0) {
          if (level == 1) {
            l1_cache_size = size * 1024;
          } else if (level == 2) {
            l2_cache_size = size * 1024;
          } else if (level == 3) {
            l3_cache_size = size * 1024;
          }
        }
      }
    }
  }
}

bool CheckCpuOnline(const std::vector<int>& cpu_ids) {
  if (cpu_ids.empty()) {
    return false;
  }
  char path[256];
  bool all_online = true;
  for (auto& id : cpu_ids) {
    snprintf(
        path, sizeof(path), "/sys/devices/system/cpu/cpu%d/online", id);
    FILE* fp = fopen(path, "rb");
    int is_online = 0;
    if (fp) {
      fscanf(fp, "%d", &is_online);
      fclose(fp);
    } else {
      printf("Failed to query the online statue of CPU id: %d\n", id);
    }
    if (is_online == 0) {
      all_online = false;
      printf("CPU id: %d is offine", id);
    }
  }
  return all_online;
}

int SetSchedAffinity(const std::vector<int>& cpu_ids) {
// set affinity for thread
#ifdef __GLIBC__
  pid_t pid = syscall(SYS_gettid);
#else
  pid_t pid = gettid();
#endif
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (auto& id : cpu_ids) {
    CPU_SET(id, &mask);
  }
  int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
  if (syscallret) {
    return -1;
  }
  return 0;
}

bool BindThreads(const std::vector<int>& cpu_ids) {
#ifdef USE_OPENMP
  int thread_num = cpu_ids.size();
  omp_set_num_threads(thread_num);
  std::vector<int> ssarets;
  for (int i = 0; i < thread_num; ++i) {
    ssarets.push_back(0);
  }
#pragma omp parallel for
  for (int i = 0; i < thread_num; i++) {
    ssarets[i] = SetSchedAffinity(cpu_ids);
  }
  for (int i = 0; i < thread_num; i++) {
    if (ssarets[i] != 0) {
      printf("Set cpu affinity failed, core id: %d\n", cpu_ids[i]);
      return false;
    }
  }
#else   // USE_OPENMP
  std::vector<int> first_cpu_id;
  first_cpu_id.push_back(cpu_ids[0]);
  int ssaret = SetSchedAffinity(first_cpu_id);
  if (ssaret != 0) {
    printf("Set cpu affinity failed, core id: %d\n", cpu_ids[0]);
    return false;
  }
#endif  // USE_OPENMP
  return true;
}
#endif  // USE_ARM_LINUX or USE_ANDROID

void ARMDevice::SetDotInfo(int argc, ...) {
  va_list arg_ptr;
  va_start(arg_ptr, argc);
  info_.dot.resize(info_.core_num);
  if (argc == 1) {
    bool flag = va_arg(arg_ptr, int) > 0;
    for (uint32_t i = 0; i < info_.core_num; ++i) {
      info_.dot[i] = flag;
    }
  } else {
    bool flag_big_core = va_arg(arg_ptr, int) > 0;
    bool flag_little_core = va_arg(arg_ptr, int) > 0;
    int big_core_num = info_.big_core_ids.size();
    int little_core_num = info_.little_core_ids.size();
    for (int i = 0; i < big_core_num; ++i) {
      info_.dot[info_.big_core_ids[i]] = flag_big_core;
    }
    for (int i = 0; i < little_core_num; ++i) {
      info_.dot[info_.little_core_ids[i]] = flag_little_core;
    }
  }
  va_end(arg_ptr);
}

void ARMDevice::SetFP16Info(int argc, ...) {
  va_list arg_ptr;
  va_start(arg_ptr, argc);
  info_.fp16.resize(info_.core_num);
  if (argc == 1) {
    bool flag = va_arg(arg_ptr, int) > 0;
    for (uint32_t i = 0; i < info_.core_num; ++i) {
      info_.fp16[i] = flag;
    }
  } else {
    bool flag_big_core = va_arg(arg_ptr, int) > 0;
    bool flag_little_core = va_arg(arg_ptr, int) > 0;
    int big_core_num = info_.big_core_ids.size();
    int little_core_num = info_.little_core_ids.size();
    for (int i = 0; i < big_core_num; ++i) {
      info_.fp16[info_.big_core_ids[i]] = flag_big_core;
    }
    for (int i = 0; i < little_core_num; ++i) {
      info_.fp16[info_.little_core_ids[i]] = flag_little_core;
    }
  }
  va_end(arg_ptr);
}

void ARMDevice::SetFP32Info(int argc, ...) {
  va_list arg_ptr;
  va_start(arg_ptr, argc);
  info_.fp32.resize(info_.core_num);
  if (argc == 1) {
    bool flag = va_arg(arg_ptr, int) > 0;
    for (uint32_t i = 0; i < info_.core_num; ++i) {
      info_.fp32[i] = flag;
    }
  } else {
    bool flag_big_core = va_arg(arg_ptr, int) > 0;
    bool flag_little_core = va_arg(arg_ptr, int) > 0;
    int big_core_num = info_.big_core_ids.size();
    int little_core_num = info_.little_core_ids.size();
    for (int i = 0; i < big_core_num; ++i) {
      info_.fp32[info_.big_core_ids[i]] = flag_big_core;
    }
    for (int i = 0; i < little_core_num; ++i) {
      info_.fp32[info_.little_core_ids[i]] = flag_little_core;
    }
  }
  va_end(arg_ptr);
}

// cache_id : 0 -> L1, 1 -> L2, 2 -> L3
void ARMDevice::SetCacheInfo(int cache_id, int argc, ...) {
  va_list arg_ptr;
  va_start(arg_ptr, argc);
  std::vector<int> cache;
  switch (cache_id) {
    case 0:
      cache = info_.L1_cache;
      break;
    case 1:
      cache = info_.L2_cache;
      break;
    case 2:
      cache = info_.L3_cache;
      break;
    default:
      break;
  }
  cache.resize(info_.core_num);
  if (argc == 1) {
    int cache_size = va_arg(arg_ptr, int);
    for (uint32_t i = 0; i < info_.core_num; ++i) {
      cache[i] = cache_size;
    }
  } else {
    int big_core_num = info_.big_core_ids.size();
    int little_core_num = info_.little_core_ids.size();
    int big_core_cache_size = va_arg(arg_ptr, int);
    int little_core_cache_size = va_arg(arg_ptr, int);
    for (int i = 0; i < big_core_num; ++i) {
      cache[info_.big_core_ids[i]] = big_core_cache_size;
    }
    for (int i = 0; i < little_core_num; ++i) {
      cache[info_.little_core_ids[i]] = little_core_cache_size;
    }
  }
  va_end(arg_ptr);
}

void ARMDevice::SetArchInfo(int argc, ...) {
  va_list arg_ptr;
  va_start(arg_ptr, argc);
  info_.archs.resize(info_.core_num);
  if (argc == 1) {
    auto arch = static_cast<ARMArch>(va_arg(arg_ptr, int));
    for (uint32_t i = 0; i < info_.core_num; ++i) {
      info_.archs[i] = arch;
    }
  } else {
    auto big_core_arch = static_cast<ARMArch>(va_arg(arg_ptr, int));
    auto little_core_arch = static_cast<ARMArch>(va_arg(arg_ptr, int));
    int big_core_num = info_.big_core_ids.size();
    int little_core_num = info_.little_core_ids.size();
    for (int i = 0; i < big_core_num; ++i) {
      info_.archs[info_.big_core_ids[i]] = big_core_arch;
    }
    for (int i = 0; i < little_core_num; ++i) {
      info_.archs[info_.little_core_ids[i]] = little_core_arch;
    }
  }
  va_end(arg_ptr);
}

bool ARMDevice::SetCPUInfoByName() {
  /* Snapdragon */
  if (info_.dev_name.find("SM8150") != std::string::npos) {  // 855
    info_.core_num = 8;
    info_.core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    info_.big_core_ids = {4, 5, 6, 7};
    info_.little_core_ids = {0, 1, 2, 3};
    info_.cluster_ids = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA76, kA55);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 2, 256 * 1024, 128 * 1024);
    SetCacheInfo(2, 1, 2048 * 1024);
    SetFP16Info(1, 1);
    SetDotInfo(1, 1);
    return true;
  } else if (info_.dev_name.find("SDM845") != std::string::npos) {  // 845
    info_.core_num = 8;
    info_.core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    info_.big_core_ids = {4, 5, 6, 7};
    info_.little_core_ids = {0, 1, 2, 3};
    info_.cluster_ids = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA75, kA55);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 2, 256 * 1024, 128 * 1024);
    SetCacheInfo(2, 1, 2048 * 1024);
    SetFP16Info(1, 1);
    return true;
  } else if (info_.dev_name.find("SDM710") != std::string::npos) {  // 710
    info_.core_num = 8;
    info_.core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    info_.big_core_ids = {6, 7};
    info_.little_core_ids = {0, 1, 2, 3, 4, 5};
    info_.cluster_ids = {1, 1, 1, 1, 1, 1, 0, 0};
    SetArchInfo(2, kA75, kA55);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 2, 256 * 1024, 128 * 1024);
    SetCacheInfo(2, 1, 1024 * 1024);
    return true;
  } else if (info_.dev_name.find("MSM8998") != std::string::npos) {  // 835
    info_.core_num = 8;
    info_.core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    info_.big_core_ids = {4, 5, 6, 7};
    info_.little_core_ids = {0, 1, 2, 3};
    info_.cluster_ids = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA73, kA53);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1,
                 2,
                 1024 * 1024,
                 /*real cache size is 2M, while that will get bad performace
                    on conv3x3s1 or gemm, set to 1M or 512K*/
                 1024 * 1024);
    return true;
  } else if (info_.dev_name.find("MSM8996") != std::string::npos) {  // 820
    info_.core_num = 4;
    info_.core_ids = {0, 1, 2, 3};
    info_.big_core_ids = {2, 3};
    info_.little_core_ids = {0, 1};
    info_.cluster_ids = {1, 1, 0, 0};
    SetArchInfo(1, kA72);
    SetCacheInfo(0, 1, 24 * 1024);
    SetCacheInfo(1, 2, 1024 * 1024, 512 * 1024);
    return true;
  } else if (info_.dev_name.find("SDM660") != std::string::npos ||
             info_.dev_name.find("SDM636") != std::string::npos) {  // 660, 636
    info_.core_num = 8;
    info_.core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    info_.big_core_ids = {4, 5, 6, 7};
    info_.little_core_ids = {0, 1, 2, 3};
    info_.cluster_ids = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(1, kA73);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 1, 1024 * 1024);
    return true;
  } else if (info_.dev_name.find("MSM8976") != std::string::npos) {  // 652,653
    info_.core_num = 8;
    info_.core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    info_.big_core_ids = {4, 5, 6, 7};
    info_.little_core_ids = {0, 1, 2, 3};
    info_.cluster_ids = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA72, kA53);
    SetCacheInfo(0, 1, 32 * 1024);
    SetCacheInfo(1, 2, 1024 * 1024, 512 * 1024);
    return true;
  } else if (info_.dev_name.find("MSM8953") != std::string::npos) {  // 625
    info_.core_num = 8;
    info_.core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    info_.big_core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    info_.little_core_ids = {};
    info_.cluster_ids = {0, 0, 0, 0, 0, 0, 0, 0};
    SetArchInfo(1, kA53);
    SetCacheInfo(0, 1, 32 * 1024);
    SetCacheInfo(1, 1, 1024 * 1024);
    return true;
  } else if (info_.dev_name.find("MSM8939") != std::string::npos) {  // 615
    info_.core_num = 8;
    info_.core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    info_.big_core_ids = {0, 1, 2, 3};
    info_.little_core_ids = {4, 5, 6, 7};
    info_.cluster_ids = {0, 0, 0, 0, 1, 1, 1, 1};
    SetArchInfo(1, kA53);
    SetCacheInfo(0, 1, 32 * 1024);
    SetCacheInfo(1, 2, 512 * 1024, 256 * 1024);
    return true;
    /* MediaTek */
  } else if (info_.dev_name.find("MT6797") !=
             std::string::npos) {  // X20/X23/X25/X27
    info_.core_num = 10;
    info_.core_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    info_.big_core_ids = {8, 9};
    info_.little_core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    info_.cluster_ids = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0};
    SetArchInfo(2, kA72, kA53);
    SetCacheInfo(0, 1, 32 * 1024);
    SetCacheInfo(1, 2, 1024 * 1024, 512 * 1024);
    return true;
  } else if (info_.dev_name.find("MT6799") != std::string::npos) {  // X30
    info_.core_num = 10;
    info_.core_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    info_.big_core_ids = {8, 9};
    info_.little_core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    info_.cluster_ids = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0};
    SetArchInfo(2, kA73, kA53);
    return true;
  } else if (info_.dev_name.find("MT6795") != std::string::npos ||
             info_.dev_name.find("MT6762") != std::string::npos ||
             info_.dev_name.find("MT6755T") != std::string::npos ||
             info_.dev_name.find("MT6755S") != std::string::npos ||
             info_.dev_name.find("MT6753") != std::string::npos ||
             info_.dev_name.find("MT6752") != std::string::npos ||
             info_.dev_name.find("MT6750") != std::string::npos) {
    // X10, P22, P15/P18, MT6753, MT6752/MT6752M, MT6750
    info_.core_num = 8;
    info_.core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    info_.big_core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    info_.little_core_ids = {};
    info_.cluster_ids = {0, 0, 0, 0, 0, 0, 0, 0};
    SetArchInfo(1, kA53);
    return true;
  } else if (info_.dev_name.find("MT6758") != std::string::npos ||
             info_.dev_name.find("MT6757") != std::string::npos ||
             info_.dev_name.find("MT6763") != std::string::npos ||
             info_.dev_name.find("MT6755M") != std::string::npos ||
             info_.dev_name.find("MT6755") !=
                 std::string::npos) {  // P30, P20/P25, P23, P10
    info_.core_num = 8;
    info_.core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    info_.big_core_ids = {4, 5, 6, 7};
    info_.little_core_ids = {0, 1, 2, 3};
    info_.cluster_ids = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(1, kA53);
    return true;
  } else if (info_.dev_name.find("MT6771") != std::string::npos) {  // P60
    info_.core_num = 8;
    info_.core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    info_.big_core_ids = {4, 5, 6, 7};
    info_.little_core_ids = {0, 1, 2, 3};
    info_.cluster_ids = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA73, kA53);
    return true;
  } else if (info_.dev_name.find("MT6765") != std::string::npos ||
             info_.dev_name.find("MT6739") != std::string::npos ||
             info_.dev_name.find("MT6738") != std::string::npos ||
             info_.dev_name.find("MT6737") !=
                 std::string::npos) {  // A22, MT6739, MT6738, MT6767
    info_.core_num = 4;
    info_.core_ids = {0, 1, 2, 3};
    info_.big_core_ids = {0, 1, 2, 3};
    info_.little_core_ids = {};
    info_.cluster_ids = {0, 0, 0, 0};
    SetArchInfo(1, kA53);
    return true;
  } else if (info_.dev_name.find("KIRIN980") != std::string::npos ||
             info_.dev_name.find("KIRIN990") !=
                 std::string::npos) {  // Kirin 980, Kirin 990
    info_.core_num = 8;
    info_.core_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    info_.big_core_ids = {4, 5, 6, 7};
    info_.little_core_ids = {0, 1, 2, 3};
    info_.cluster_ids = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA76, kA55);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 2, 512 * 1024, 128 * 1024);
    SetCacheInfo(2, 1, 4096 * 1024);
    SetFP16Info(1, 1);
    SetDotInfo(1, 1);
    return true;
  }
  return false;
}

void ARMDevice::SetCPUInfoByProb() {
#if defined(USE_ARM_LINUX) || defined(USE_ANDROID)
  // get big.LITTLE cores by sorting CPU frequency
  SortCpuidByMaxFreq(info_.max_freqs, &info_.core_ids, &info_.cluster_ids);
  info_.big_core_ids.clear();
  info_.little_core_ids.clear();
  for (size_t i = 0; i < info_.cluster_ids.size(); ++i) {
    if (info_.cluster_ids[i] == 0) {
      info_.big_core_ids.push_back(info_.core_ids[i]);
    } else {
      info_.little_core_ids.push_back(info_.core_ids[i]);
    }
  }
  // get l1, l2, l3 cache size for each core
  for (uint32_t i = 0; i < info_.core_num; i++) {
    GetCpuCacheSize(i, info_.L1_cache[i], info_.L2_cache[i], info_.L3_cache[i]);
  }
#endif  // LITE_WITH_LINUX
}

int ARMDevice::Setup() {
  info_.core_num = GetCpuNum();
  info_.mem_size = GetMemSize();
  GetCpuArch(info_.archs, info_.core_num);
  // set defalut CPU info
  SetCacheInfo(0, 1, DEFAULT_L1_CACHE_SIZE);
  SetCacheInfo(1, 1, DEFAULT_L2_CACHE_SIZE);
  SetCacheInfo(2, 1, DEFAULT_L3_CACHE_SIZE);
  SetFP32Info(1, 1);
  SetFP16Info(1, 0);
  SetDotInfo(1, 0);
  info_.max_freqs.resize(info_.core_num);
  info_.min_freqs.resize(info_.core_num);
#if defined(USE_ARM_LINUX) || defined(USE_ANDROID)
  // get max&min freq
  for (uint32_t i = 0; i < info_.core_num; ++i) {
    int max_freq = GetMaxFreqKhz(i);
    int min_freq = GetMinFreqKhz(i);
    info_.max_freqs[i] = max_freq / 1000;
    info_.min_freqs[i] = min_freq / 1000;
  }
  // get cache size and big.LITTLE core ids
  info_.dev_name = GetCpuName();
  if (!SetCPUInfoByName()) {
    SetCPUInfoByProb();
  }
#else
#ifdef USE_IOS
  info_.dev_name = "Apple";
#else
  info_.dev_name = "Unknown";
#endif
  info_.core_ids.resize(info_.core_num);
  info_.cluster_ids.resize(info_.core_num);
  info_.big_core_ids.resize(info_.core_num);
  for (int i = 0; i < info_.core_num; ++i) {
    info_.max_freqs[i] = 1000000;
    info_.min_freqs[i] = 1000000;
    info_.cluster_ids[i] = 0;
    info_.core_ids[i] = i;
    info_.big_core_ids[i] = i;
  }
#endif
  // output info
  printf("ARM multiprocessors name: %s\n", info_.dev_name.c_str());
  printf("ARM multiprocessors number: %d\n", info_.core_num);
  printf("Total memory: %dKB\n", info_.mem_size);
  for (uint32_t i = 0; i < info_.core_num; ++i) {
    printf("ARM multiprocessors ID: %d, max freq: %d, min freq: %d, cluster ID: %d, CPU ARCH: A%d\n",
            info_.core_ids[i], info_.max_freqs[i], info_.min_freqs[i],
            info_.cluster_ids[info_.core_ids[i]], static_cast<int>(info_.archs[i]));
  }
  for (uint32_t i = 0; i < info_.core_num; ++i) {
    printf("core id %d: L1Cache Size: %dKB, L2Cache Size: %dKB, L3Cache Size: %dKB\n",
            i, info_.L1_cache[i] / 1024, info_.L2_cache[i] / 1024, info_.L3_cache[i] / 1024);
  }
  return 0;
}

}  // namespace arm
}  // namespace onnxruntime
