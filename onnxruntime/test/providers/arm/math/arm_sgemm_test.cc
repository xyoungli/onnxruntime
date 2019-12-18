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

#include <gtest/gtest.h>
#include "test/data_utils.h"
#include "test/naive_math_impl.h"
#include "test/timer.h"
#include "core/providers/arm/funcs/sgemm.h"
#include "core/providers/arm/funcs/sgemm_packed.h"

const bool precision_test_basic = true;
const bool precision_test_full = false;
const bool performance_test = true;

namespace onnxruntime {
namespace test {

bool TestSgemm(bool tra, bool trb,
               int m, int n, int k,
               float alpha, int lda,
               int ldb,
               float beta, int ldc,
               bool with_bias, bool with_relu,
               int cls, int ths,
               int warmup_iter=0, int repeats=1, bool check_result=true) {

  ARMExecutionProviderInfo info;
  info.threads = ths;
  info.mode = static_cast<PowerMode>(cls);
  auto provider = onnxruntime::make_unique<ARMExecutionProvider>(info);
  auto alloc = provider->GetAllocator(0, OrtMemTypeDefault);

  int size_a = tra ? k * lda : m * lda;
  int size_b = trb ? n * ldb : k * ldb;
  int size_c = m * ldc;

  auto da = static_cast<float *>(alloc->Alloc(size_a * sizeof(float)));
  auto db = static_cast<float *>(alloc->Alloc(size_b * sizeof(float)));
  auto dbias = static_cast<float *>(alloc->Alloc(m * sizeof(float)));
  auto dc = static_cast<float *>(alloc->Alloc(size_c * sizeof(float)));
  auto dc_basic = static_cast<float *>(alloc->Alloc(size_c * sizeof(float)));
  auto dc_backup = static_cast<float *>(alloc->Alloc(size_c * sizeof(float)));

  fill_data_rand(da, -1.f, 1.f, size_a);
  fill_data_rand(db, -1.f, 1.f, size_b);
  fill_data_rand(dbias, -1.f, 1.f, m);
  fill_data_rand(dc, -1.f, 1.f, size_c);

  memcpy(dc_basic, dc, sizeof(float) * size_c);
  memcpy(dc_backup, dc, sizeof(float) * size_c);

  if (check_result) {
    basic_gemm(tra, trb,
               m, n, k,
               alpha, da, lda,
               db, ldb,
               beta, dc_basic, ldc,
               dbias, with_bias, with_relu);
  }
  Timer t0;
  //! compute
  double ops = 2.0 * m * n * k;
  //! prepack
  int hblock = arm::GetSgemmHblock(provider.get());
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  auto alloc_ptr = provider->GetAllocator(0, OrtMemTypeDefault);
  auto packed_A_ptr = static_cast<float*>(
          alloc_ptr->Alloc(m_roundup * k * sizeof(float)));

  arm::PrepackA(packed_A_ptr, da, alpha, lda, 0, m, 0, k, tra, provider.get());
  for (int j = 0; j < warmup_iter; ++j) {
    arm::SgemmPrepack(trb,
                      m, n, k,
                      packed_A_ptr,
                      db, ldb,
                      beta, dc, ldc,
                      dbias, with_bias, with_relu, provider.get());
  }

  for (int i = 0; i < repeats; ++i) {
    if (i == repeats - 1) {
      memcpy(dc, dc_backup, sizeof(float) * m * ldc);
    }
    t0.Start();
    arm::SgemmPrepack(trb,
                      m, n, k,
                      packed_A_ptr,
                      db, ldb,
                      beta, dc, ldc,
                      dbias, with_bias, with_relu, provider.get());
    t0.Stop();
  }
  std::cout << "SgemmPrepack , transA: " << (tra ? "true" : "false")
            << ", transB: " << (trb ? "true" : "false")
            << ", M: " << m << ", N: " << n << ", K: " << k
            << ", alpha: " << alpha <<  ", beta: " << beta
            << ", lda: " << lda << ", ldb: " << ldb << ", ldc: " << ldc
            << ", bias: " << (with_bias ? "true" : "false")
            << ", relu: " << (with_relu ? "true" : "false")
            << ", power_mode: " << cls << ", threads: " << ths
            << ", GOPS: " << ops * 1e-9f << "GOPS"
            << ", avg time: " << t0.LapTimes().Avg() << "ms"
            << ", min time: " << t0.LapTimes().Min() << "ms"
            << ", mean GOPs: " << ops * 1e-6f / t0.LapTimes().Avg() << "GOPs"
            << ", max GOPs: " << ops * 1e-6f / t0.LapTimes().Min() << " GOPs\n";

  if (check_result) {
    double max_ratio = 0;
    double max_diff = 0;
    check_precision(dc_basic, dc, max_ratio, max_diff, size_c);
    std::cout << "compare result, max diff: " << max_diff
              << ", max ratio: " << max_ratio << std::endl;
    if (std::abs(max_ratio) > 1e-4f && std::abs(max_diff) > 5e-5f) {
      auto data_diff = static_cast<float *>(alloc->Alloc(size_c * sizeof(float)));
      compute_data_diff(dc_basic, dc, data_diff, m);
      std::cout << "basic result: \n";
      print_data(dc_basic, size_c, ldc);
      std::cout << "saber result: \n";
      print_data(dc, size_c, ldc);
      std::cout << "diff result: \n";
      print_data(data_diff, size_c, ldc);
      return false;
    }
  }
  return true;
}

TEST(TestSgemm, TestSgemmPrecisionFull) {
  if (precision_test_full) {
    std::cout << "run basic sgemm test\n";
    for (auto &m : {1, 3, 8, 32, 397}) {
      for (auto &n : {1, 3, 13, 141, 512, 789}) {
        for (auto &k : {1, 3, 8, 59, 234}) {
          for (auto &tra : {false, true}) {
            for (auto &trb : {false, true}) {
              for (auto &alpha : {1.f, 0.5f}) {
                for (auto &beta : {0.f, 0.5f}) {
                  for (auto &offset : {0, 10}) {
                    for (auto &has_bias : {false, true}) {
                      for (auto &has_relu : {false, true}) {
                        for (auto &th : {1, 2, 4}) {
                          int lda = k + offset;
                          if (tra) {
                            lda = m + offset;
                          }
                          int ldb = n + offset;
                          if (trb) {
                            ldb = k + offset;
                          }
                          int ldc = n + offset;
                          auto flag = TestSgemm(tra, trb,
                                                 m, n, k,
                                                 alpha, lda, ldb, beta, ldc,
                                                 has_bias, has_relu,
                                                 0, th, 0, 1, true);
                          if (flag) {
                            std::cout << "test m = " << m << ", n=" << n
                                      << ", k=" << k
                                      << ", bias: " << (has_bias ? "true" : "false")
                                      << ", relu: " << (has_relu ? "true" : "false")
                                      << ", trans A: " << (tra ? "true" : "false")
                                      << ", trans B: " << (trb ? "true" : "false")
                                      << " passed\n";

                          } else {
                            std::cout << "test m = " << m << ", n=" << n
                                      << ", k=" << k
                                      << ", bias: " << (has_bias ? "true" : "false")
                                      << ", relu: " << (has_relu ? "true" : "false")
                                      << ", trans A: " << (tra ? "true" : "false")
                                      << ", trans B: " << (trb ? "true" : "false")
                                      << " failed\n";
                            EXPECT_TRUE(false);
                            return;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(TestSgemm, TestSgemmPrecision) {
  if (precision_test_basic) {
    std::cout << "run basic sgemm test\n";
    for (auto &m : {1, 3, 13, 32}) {
      for (auto &n : {1, 3, 13, 32}) {
        for (auto &k : {1, 3, 13, 32}) {
          for (auto &tra : {false, true}) {
            for (auto &trb : {false, true}) {
              for (auto &alpha : {1.f, 0.5f}) {
                for (auto &beta : {0.f, 0.5f}) {
                  for (auto &offset : {0, 10}) {
                    for (auto &has_bias : {false, true}) {
                      for (auto &has_relu : {false, true}) {
                        for (auto &th : {1, 2, 4}) {
                          int lda = k + offset;
                          if (tra) {
                            lda = m + offset;
                          }
                          int ldb = n + offset;
                          if (trb) {
                            ldb = k + offset;
                          }
                          int ldc = n + offset;
                          auto flag = TestSgemm(tra, trb,
                                                m, n, k,
                                                alpha, lda, ldb, beta, ldc,
                                                has_bias, has_relu,
                                                0, th, 0, 1, true);
                          if (flag) {
                            std::cout << "test m = " << m << ", n=" << n
                                      << ", k=" << k
                                      << ", bias: " << (has_bias ? "true" : "false")
                                      << ", relu: " << (has_relu ? "true" : "false")
                                      << ", trans A: " << (tra ? "true" : "false")
                                      << ", trans B: " << (trb ? "true" : "false")
                                      << " passed\n";

                          } else {
                            std::cout << "test m = " << m << ", n=" << n
                                      << ", k=" << k
                                      << ", bias: " << (has_bias ? "true" : "false")
                                      << ", relu: " << (has_relu ? "true" : "false")
                                      << ", trans A: " << (tra ? "true" : "false")
                                      << ", trans B: " << (trb ? "true" : "false")
                                      << " failed\n";
                            EXPECT_TRUE(false);
                            return;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(TestSgemm, TestSgemmPerformance) {
  const int m = 512;
  const int n = 512;
  const int k = 512;
  for (auto& tra : {false, true}) {
    for (auto& trb: {false, true}) {
      for (auto& th : {1, 2, 4}) {
        int lda = tra? m : k;
        int ldb = trb? k : n;
        int ldc = n;
        TestSgemm(tra, trb, m, n, k,
                1.f, lda, ldb, 0.f, ldc,
                false, false, 0, th, 10, 50, false);
      }
    }
  }
}

}  //  namespace test
}  //  namespace onnxruntime