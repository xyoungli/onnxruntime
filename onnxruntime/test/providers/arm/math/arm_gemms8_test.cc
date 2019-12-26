#include <gtest/gtest.h>
#include "test/data_utils.h"
#include "test/naive_math_impl.h"
#include "test/timer.h"
#include "core/providers/arm/funcs/gemm_int8.h"
#include "core/providers/arm/funcs/calib.h"

const bool precision_test = true;
const bool performance_test = true;

namespace onnxruntime {
namespace test {

bool TestGemms8(bool tra, bool trb,
                int m, int n, int k,
                bool with_bias, bool with_relu,
                int cls, int ths,
                int warmup_iter=0, int repeats=1, bool check_result=true) {
  ARMExecutionProviderInfo info;
  info.threads = ths;
  info.mode = static_cast<PowerMode>(cls);
  auto provider = onnxruntime::make_unique<ARMExecutionProvider>(info);
  auto alloc = provider->GetAllocator(0, OrtMemTypeDefault);

  std::cout << "Gemms8 , transA: " << (tra ? "true" : "false")
            << ", transB: " << (trb ? "true" : "false")
            << ", M: " << m << ", N: " << n << ", K: " << k
            << ", bias: " << (with_bias ? "true" : "false")
            << ", relu: " << (with_relu ? "true" : "false")
            << ", power_mode: " << cls << ", threads: " << ths;

  int size_a = m * k;
  int size_b = n * k;
  int size_c = m * n;

  Buffer ta(alloc);
  Buffer ta_pack(alloc);
  Buffer tb(alloc);
  Buffer tbias(alloc);
  Buffer tbias_int8(alloc);

  Buffer tc_int8(alloc);
  Buffer tc_fp32(alloc);

  Buffer tc_prepack_int8(alloc);
  Buffer tc_prepack_fp32(alloc);

  Buffer tc_basic_int8(alloc);
  Buffer tc_basic_fp32(alloc);

  ta.ReAlloc(size_a * sizeof(int8_t));
  tb.ReAlloc(size_b * sizeof(int8_t));
  tbias.ReAlloc(m * sizeof(float));
  tbias_int8.ReAlloc(m * sizeof(float));

  tc_int8.ReAlloc(size_c * sizeof(int8_t));
  tc_fp32.ReAlloc(size_c * sizeof(float));

  tc_prepack_int8.ReAlloc(size_c * sizeof(int8_t));
  tc_prepack_fp32.ReAlloc(size_c * sizeof(float));

  tc_basic_int8.ReAlloc(size_c * sizeof(int8_t));
  tc_basic_fp32.ReAlloc(size_c * sizeof(float));


  auto da = ta.MutableData<int8_t>();
  auto db = tb.MutableData<int8_t>();
  auto dbias = with_bias? tbias.MutableData<float>() : nullptr;
  auto dbias_int8_out = with_bias? tbias_int8.MutableData<float>() : nullptr;

  auto dc_int8 = tc_int8.MutableData<int8_t>();
  auto dc_fp32 = tc_fp32.MutableData<float>();

  auto dc_prepack_int8 = tc_prepack_int8.MutableData<int8_t>();
  auto dc_prepack_fp32 = tc_prepack_fp32.MutableData<float>();

  auto dc_basic_int8 = tc_basic_int8.MutableData<int8_t>();
  auto dc_basic_fp32 = tc_basic_fp32.MutableData<float>();

  fill_data_rand(da, static_cast<int8_t>(-127), static_cast<int8_t>(127), size_a);
  fill_data_rand(db, static_cast<int8_t>(-127), static_cast<int8_t>(127), size_b);
  if (with_bias) {
    fill_data_rand(dbias, -1.f, 1.f, m);
  }

  std::vector<float> scale_a(static_cast<size_t>(m), 1.f / 127);
  std::vector<float> scale_b = {1.f / 127};
  std::vector<float> scale_c = {k / 127.f};
  std::vector<float> scale_merge_fp32(static_cast<size_t>(m));
  std::vector<float> scale_merge_int8(static_cast<size_t>(m));
  for (int j = 0; j < m; ++j) {
    scale_merge_fp32[j] = scale_a[j] * scale_b[0];
    scale_merge_int8[j] = scale_merge_fp32[j] / scale_c[0];
  }

  int lda = tra ? m : k;
  int ldb = trb ? k : n;
  int ldc = n;

  if (check_result) {
    auto da_fp32 = static_cast<float *>(alloc->Alloc(size_a * sizeof(float)));
    auto db_fp32 = static_cast<float *>(alloc->Alloc(size_b * sizeof(float)));

    arm::funcs::Int8ToFp32(da, da_fp32, scale_a.data(), 1, 1, size_a);
    arm::funcs::Int8ToFp32(db, db_fp32, scale_b.data(), 1, 1, size_b);
    basic_gemm(tra, trb,
               m, n, k,
               1.f, da_fp32, lda,
               db_fp32, ldb,
               0.f, dc_basic_fp32, ldc,
               dbias, with_bias, with_relu);
    arm::funcs::Fp32ToInt8(dc_basic_fp32, dc_basic_int8, scale_c.data(), 1, 1, size_c);
    alloc->Free(da_fp32);
    alloc->Free(db_fp32);
  }
  //! compute
  double ops = 2.0 * m * n * k;
  //! prepack
  int hblock = arm::funcs::GetHblockInt8(provider.get());
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int round_up_k = 4 * ((k + 3) / 4);
  ta_pack.ReAlloc(m_roundup * round_up_k * sizeof(int8_t));
  auto packed_A_ptr = ta_pack.MutableData<int8_t>();
  arm::funcs::PrepackAInt8(packed_A_ptr, da, lda, 0, m, 0, k, tra, provider.get());
  /// warmup
  for (int j = 0; j < warmup_iter; ++j) {
    arm::funcs::GemmPrepackInt8(packed_A_ptr, db, dbias, dc_prepack_fp32,
                                m, n, k, with_bias, with_relu,
                                trb, scale_merge_fp32.data(), provider.get());
  }

  // packed fp32
  Timer t0;
  for (int i = 0; i < repeats; ++i) {
    t0.Start();
    arm::funcs::GemmPrepackInt8(packed_A_ptr, db, dbias, dc_prepack_fp32,
                                m, n, k, with_bias, with_relu,
                                trb, scale_merge_fp32.data(), provider.get());
    t0.Stop();
  }

  /// convert bias to fit int8 output
  if (with_bias) {
    for (int l = 0; l < m; ++l) {
      dbias_int8_out[l] = dbias[l] / scale_c[0];
    }
  }
  // packed int8
  Timer t1;
  for (int i = 0; i < repeats; ++i) {
    t1.Start();
    arm::funcs::GemmPrepackInt8(packed_A_ptr, db, dbias_int8_out, dc_prepack_int8,
                                m, n, k, with_bias, with_relu,
                                trb, scale_merge_int8.data(), provider.get());
    t1.Stop();
  }

  // packed fp32
  Timer t2;
  for (int i = 0; i < repeats; ++i) {
    t2.Start();
    arm::funcs::GemmInt8(tra, trb, m, n, k, da, db, dc_fp32,
                         dbias, with_bias, with_relu,
                         scale_merge_fp32.data(), provider.get());
    t2.Stop();
  }

  // int8
  Timer t3;
  for (int i = 0; i < repeats; ++i) {
    t3.Start();
    arm::funcs::GemmInt8(tra, trb, m, n, k, da, db, dc_int8,
                         dbias_int8_out, with_bias, with_relu,
                         scale_merge_int8.data(), provider.get());
    t3.Stop();
  }

  std::cout << ", GOPS: " << ops * 1e-9f << "GOPS\n"
            << "GemmPrepack Fp32 avg time: " << t0.LapTimes().Avg() << "ms"
            << ", min time: " << t0.LapTimes().Min() << "ms"
            << ", mean GOPs: " << ops * 1e-6f / t0.LapTimes().Avg() << "GOPs"
            << ", max GOPs: " << ops * 1e-6f / t0.LapTimes().Min() << " GOPs\n"
            << "GemmPrepack Int8 avg time: " << t1.LapTimes().Avg() << "ms"
            << ", min time: " << t1.LapTimes().Min() << "ms"
            << ", mean GOPs: " << ops * 1e-6f / t1.LapTimes().Avg() << "GOPs"
            << ", max GOPs: " << ops * 1e-6f / t1.LapTimes().Min() << " GOPs\n"
            << "Gemm Fp32 avg time: " << t2.LapTimes().Avg() << "ms"
            << ", min time: " << t2.LapTimes().Min() << "ms"
            << ", mean GOPs: " << ops * 1e-6f / t2.LapTimes().Avg() << "GOPs"
            << ", max GOPs: " << ops * 1e-6f / t2.LapTimes().Min() << " GOPs\n"
            << "Gemm Int8 avg time: " << t3.LapTimes().Avg() << "ms"
            << ", min time: " << t3.LapTimes().Min() << "ms"
            << ", mean GOPs: " << ops * 1e-6f / t3.LapTimes().Avg() << "GOPs"
            << ", max GOPs: " << ops * 1e-6f / t3.LapTimes().Min() << " GOPs\n";

  if (check_result) {
    /// fp32
    if (!CheckFp32(dc_basic_fp32, dc_prepack_fp32, size_c, ldc)) {
      std::cout << "check prepacked gemms8 fp32 out failed\n";
      return false;
    }
    if (!CheckFp32(dc_basic_fp32, dc_fp32, size_c, ldc)) {
      std::cout << "check gemms8 fp32 out failed\n";
      return false;
    }
    /// int8
    if (!CheckInt8(dc_basic_int8, dc_prepack_int8, size_c, ldc, 0.01f, 10)) {
      std::cout << "check prepacked gemms8 int8 out failed\n";
      return false;
    }
    if (!CheckInt8(dc_basic_int8, dc_int8, size_c, ldc, 0.01f, 10)) {
      std::cout << "check gemms8 int8 out failed\n";
      return false;
    }
  }
  return true;
}

TEST(TestLiteGemmInt8, gemm_prepacked_int8) {
  if (precision_test) {
    std::cout << "run basic sgemm test\n";
    for (auto& m : {1, 3, 8, 32, 397}) {
      for (auto& n : {1, 3, 13, 141, 512, 789}) {
        for (auto& k : {1, 3, 8, 59, 234}) {
          for (auto& tra : {false, true}) {
            for (auto& trb : {false, true}) {
              for (auto& has_bias : {false, true}) {
                for (auto& has_relu : {false, true}) {
                  for (auto& th : {1, 2, 4}) {
                    auto flag = TestGemms8(tra, trb,
                                           m, n, k,
                                           has_bias, has_relu,
                                           0, th);
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

TEST(TestSgemm, TestSgemmPerformanceCubic) {
  const int m = 512;
  const int n = 512;
  const int k = 512;
  for (auto& tra : {false, true}) {
    for (auto& trb: {false, true}) {
      for (auto& th : {1, 2, 4}) {
        TestGemms8(tra, trb,
                   m, n, k,
                   false, false, 0, th, 10, 50, false);
      }
    }
  }
}

}  //  namespace test
}  //  namespace onnxruntime
