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

  std::cout << "Sgemm , transA: " << (tra ? "true" : "false")
            << ", transB: " << (trb ? "true" : "false")
            << ", M: " << m << ", N: " << n << ", K: " << k
            << ", alpha: " << alpha <<  ", beta: " << beta
            << ", lda: " << lda << ", ldb: " << ldb << ", ldc: " << ldc
            << ", bias: " << (with_bias ? "true" : "false")
            << ", relu: " << (with_relu ? "true" : "false")
            << ", power_mode: " << cls << ", threads: " << ths;

  int size_a = tra ? k * lda : m * lda;
  int size_b = trb ? n * ldb : k * ldb;
  int size_c = m * ldc;

  Buffer ta(alloc);
  Buffer ta_pack(alloc);
  Buffer tb(alloc);
  Buffer tbias(alloc);

  Buffer tc(alloc);
  Buffer tc_prepack(alloc);

  Buffer tc_basic(alloc);
  Buffer tc_basic_backup(alloc);

  ta.ReAlloc(size_a * sizeof(float));
  tb.ReAlloc(size_b * sizeof(float));
  tbias.ReAlloc(m * sizeof(float));

  tc.ReAlloc(size_c * sizeof(float));
  tc_prepack.ReAlloc(size_c * sizeof(float));

  tc_basic.ReAlloc(size_c * sizeof(float));
  tc_basic_backup.ReAlloc(size_c * sizeof(float));

  auto da = ta.MutableData<float>();
  auto db = tb.MutableData<float>();
  auto dbias = with_bias? tbias.MutableData<float>() : nullptr;
  auto dc = tc.MutableData<float>();
  auto dc_prepack = tc_prepack.MutableData<float>();
  auto dc_basic = tc_basic.MutableData<float>();
  auto dc_backup = tc_basic_backup.MutableData<float>();

  fill_data_rand(da, -1.f, 1.f, size_a);
  fill_data_rand(db, -1.f, 1.f, size_b);
  fill_data_rand(dc, -1.f, 1.f, size_c);
  if (with_bias) {
    fill_data_rand(dbias, -1.f, 1.f, m);
  }

  memcpy(dc_prepack, dc, sizeof(float) * size_c);
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
  int hblock = arm::funcs::GetSgemmHblock(provider.get(), m);
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  ta_pack.ReAlloc(m_roundup * k * sizeof(float));
  auto packed_A_ptr = ta_pack.MutableData<float>();
  arm::funcs::PrepackA(packed_A_ptr, da, alpha, lda, 0, m, 0, k, tra, provider.get());
  for (int j = 0; j < warmup_iter; ++j) {
    arm::funcs::SgemmPrepack(trb,
                             m, n, k,
                             packed_A_ptr,
                             db, ldb,
                             beta, dc, ldc,
                             dbias, with_bias, with_relu, provider.get());
  }

  for (int i = 0; i < repeats; ++i) {
    if (i == repeats - 1) {
      memcpy(dc_prepack, dc_backup, sizeof(float) * m * ldc);
    }
    t0.Start();
    arm::funcs::SgemmPrepack(trb,
                             m, n, k,
                             packed_A_ptr,
                             db, ldb,
                             beta, dc_prepack, ldc,
                             dbias, with_bias, with_relu, provider.get());
    t0.Stop();
  }

  Timer t1;
  for (int i = 0; i < repeats; ++i) {
    if (i == repeats - 1) {
      memcpy(dc, dc_backup, sizeof(float) * m * ldc);
    }
    t1.Start();
    arm::funcs::Sgemm(tra, trb,
                      m, n, k,
                      alpha, da, lda,
                      db, ldb,
                      beta, dc, ldc,
                      dbias, with_bias, with_relu, provider.get());
    t1.Stop();
  }

  std::cout << ", GOPS: " << ops * 1e-9f << "GOPS\n"
            << "SgemmPrepack avg time: " << t0.LapTimes().Avg() << "ms"
            << ", min time: " << t0.LapTimes().Min() << "ms"
            << ", mean GOPs: " << ops * 1e-6f / t0.LapTimes().Avg() << "GOPs"
            << ", max GOPs: " << ops * 1e-6f / t0.LapTimes().Min() << " GOPs\n"
            << "Sgemm avg time: " << t1.LapTimes().Avg() << "ms"
            << ", min time: " << t1.LapTimes().Min() << "ms"
            << ", mean GOPs: " << ops * 1e-6f / t1.LapTimes().Avg() << "GOPs"
            << ", max GOPs: " << ops * 1e-6f / t1.LapTimes().Min() << " GOPs\n";

  if (check_result) {
    if (!CheckFp32(dc_basic, dc_prepack, size_c, ldc)) {
      std::cout << "check prepacked sgemm failed\n";
      return false;
    }
    if (!CheckFp32(dc_basic, dc, size_c, ldc)) {
      std::cout << "check sgemm failed\n";
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

TEST(TestSgemm, TestSgemmPerformanceCubic) {
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

TEST(TestSgemm, TestSgemmPerformanceNarrow) {
  const int m = 1;
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

TEST(TestSgemm, TestSgemmPerformanceWide) {
  const int m = 512;
  const int n = 1;
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