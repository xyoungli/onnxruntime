#include <gtest/gtest.h>
#include "test/data_utils.h"
#include "test/naive_math_impl.h"
#include "test/timer.h"
#include "core/providers/arm/funcs/gemv_int8.h"
#include "core/providers/arm/funcs/calib.h"

const bool precision_test = true;
const bool performance_test = true;

namespace onnxruntime {
namespace test {

bool TestGemvs8(bool trans,
                int m, int n,
                bool with_bias, bool with_relu,
                int cls, int ths,
                int warmup_iter=0, int repeats=1, bool check_result=true) {
  ARMExecutionProviderInfo info;
  info.threads = ths;
  info.mode = static_cast<PowerMode>(cls);
  auto provider = onnxruntime::make_unique<ARMExecutionProvider>(info);
  auto alloc = provider->GetAllocator(0, OrtMemTypeDefault);

  std::cout << "Gemvs8 , trans: " << (trans ? "true" : "false")
            << ", M: " << m << ", N: "
            << ", bias: " << (with_bias ? "true" : "false")
            << ", relu: " << (with_relu ? "true" : "false")
            << ", power_mode: " << cls << ", threads: " << ths;

  int size_a = m * n;
  int size_b = n;
  int size_c = m;

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

  tc_basic_int8.ReAlloc(size_c * sizeof(int8_t));
  tc_basic_fp32.ReAlloc(size_c * sizeof(float));


  auto da = ta.MutableData<int8_t>();
  auto db = tb.MutableData<int8_t>();
  auto dbias = with_bias? tbias.MutableData<float>() : nullptr;
  auto dbias_int8_out = with_bias? tbias_int8.MutableData<float>() : nullptr;

  auto dc_int8 = tc_int8.MutableData<int8_t>();
  auto dc_fp32 = tc_fp32.MutableData<float>();

  auto dc_basic_int8 = tc_basic_int8.MutableData<int8_t>();
  auto dc_basic_fp32 = tc_basic_fp32.MutableData<float>();

  fill_data_rand(da, static_cast<int8_t>(-127), static_cast<int8_t>(127), size_a);
  fill_data_rand(db, static_cast<int8_t>(-127), static_cast<int8_t>(127), size_b);
  if (with_bias) {
    fill_data_rand(dbias, -1.f, 1.f, m);
  }

  std::vector<float> scale_a(static_cast<size_t>(m), 1.f / 127);
  std::vector<float> scale_b = {1.f / 127};
  std::vector<float> scale_c = {n / 127.f};
  std::vector<float> scale_merge_fp32(static_cast<size_t>(m));
  std::vector<float> scale_merge_int8(static_cast<size_t>(m));
  for (int j = 0; j < m; ++j) {
    scale_merge_fp32[j] = scale_a[j] * scale_b[0];
    scale_merge_int8[j] = scale_merge_fp32[j] / scale_c[0];
  }

  if (check_result) {
    auto da_fp32 = static_cast<float *>(alloc->Alloc(size_a * sizeof(float)));
    auto db_fp32 = static_cast<float *>(alloc->Alloc(size_b * sizeof(float)));

    arm::funcs::Int8ToFp32(da, da_fp32, scale_a.data(), 1, 1, size_a);
    arm::funcs::Int8ToFp32(db, db_fp32, scale_b.data(), 1, 1, size_b);
    basic_gemv(trans,
               m, n,
               1.f, da_fp32, trans? m : n,
               db_fp32, 1,
               0.f, dc_basic_fp32, 1,
               dbias, with_bias, with_relu);
    arm::funcs::Fp32ToInt8(dc_basic_fp32, dc_basic_int8, scale_c.data(), 1, 1, size_c);
    alloc->Free(da_fp32);
    alloc->Free(db_fp32);
  }
  double ops = 2.0 * m * n;
  //! compute
  /// warmup
  for (int j = 0; j < warmup_iter; ++j) {
    arm::funcs::GemvInt8(da, db, dc_fp32, trans, m, n, scale_merge_fp32.data(),
            with_bias, dbias, with_relu, provider.get());
  }
  // fp32
  Timer t0;
  for (int i = 0; i < repeats; ++i) {
    t0.Start();
    arm::funcs::GemvInt8(da, db, dc_fp32, trans, m, n, scale_merge_fp32.data(),
                         with_bias, dbias, with_relu, provider.get());
    t0.Stop();
  }

  /// convert bias to fit int8 output
  if (with_bias) {
    for (int l = 0; l < m; ++l) {
      dbias_int8_out[l] = dbias[l] / scale_c[0];
    }
  }
  // int8
  Timer t1;
  for (int i = 0; i < repeats; ++i) {
    t1.Start();
    arm::funcs::GemvInt8(da, db, dc_int8, trans, m, n, scale_merge_int8.data(),
                         with_bias, dbias_int8_out, with_relu, provider.get());
    t1.Stop();
  }

  std::cout << ", GOPS: " << ops * 1e-9f << "GOPS\n"
            << "Gemvs8 Fp32 avg time: " << t0.LapTimes().Avg() << "ms"
            << ", min time: " << t0.LapTimes().Min() << "ms"
            << ", mean GOPs: " << ops * 1e-6f / t0.LapTimes().Avg() << "GOPs"
            << ", max GOPs: " << ops * 1e-6f / t0.LapTimes().Min() << " GOPs\n"
            << "Gemvs8 Int8 avg time: " << t1.LapTimes().Avg() << "ms"
            << ", min time: " << t1.LapTimes().Min() << "ms"
            << ", mean GOPs: " << ops * 1e-6f / t1.LapTimes().Avg() << "GOPs"
            << ", max GOPs: " << ops * 1e-6f / t1.LapTimes().Min() << " GOPs\n";

  if (check_result) {
    /// fp32
    if (!CheckFp32(dc_basic_fp32, dc_fp32, size_c, size_c)) {
      std::cout << "check gemvs8 fp32 out failed\n";
      return false;
    }
    if (!CheckInt8(dc_basic_int8, dc_int8, size_c, size_c, 0.01f, 10)) {
      std::cout << "check gemvs8 int8 out failed\n";
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
          for (auto& trans : {false}) {
            for (auto& has_bias : {false, true}) {
              for (auto& has_relu : {false, true}) {
                for (auto& th : {1, 2, 4}) {
                  auto flag = TestGemvs8(trans, m, n, has_bias, has_relu, 0, th);
                  if (flag) {
                    std::cout << "test m = " << m << ", n=" << n
                              << ", k=" << k
                              << ", bias: " << (has_bias ? "true" : "false")
                              << ", relu: " << (has_relu ? "true" : "false")
                              << ", trans: " << (trans ? "true" : "false")
                              << " passed\n";
                  } else {
                    std::cout << "test m = " << m << ", n=" << n
                              << ", k=" << k
                              << ", bias: " << (has_bias ? "true" : "false")
                              << ", relu: " << (has_relu ? "true" : "false")
                              << ", trans: " << (trans ? "true" : "false")
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

TEST(TestSgemm, TestSgemmPerformanceCubic) {
  const int m = 512;
  const int n = 512;
  for (auto& trans : {false}) {
    for (auto& th : {1, 2, 4}) {
      TestGemvs8(trans, m, n, false, false, 0, th, 10, 50, false);
    }
  }
}

}  //  namespace test
}  //  namespace onnxruntime
