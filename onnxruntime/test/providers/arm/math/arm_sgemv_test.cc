#include <gtest/gtest.h>
#include "test/data_utils.h"
#include "test/naive_math_impl.h"
#include "test/timer.h"
#include "core/providers/arm/funcs/sgemv.h"

const bool precision_test = true;
const bool performance_test = true;

namespace onnxruntime {
namespace test {

bool TestSgemv(bool tra, int m, int n, float alpha, int lda, float beta,
               bool has_bias, bool has_relu, int cls, int ths,
               int warmup_iter=0, int repeats=1, bool check_result=true) {

  std::cout << "sgemv, transA: " << (tra ? "true" : "false")
            << ", M: " << m << ", N: " << n
            << ", alpha: " << alpha << ", lda: " << lda
            << ", beta: " << beta
            << ", bias: " << (has_bias ? "true" : "false")
            << ", relu: " << (has_relu ? "true" : "false")
            << ", cluster: " << cls << ", threads: " << ths;

  ARMExecutionProviderInfo info;
  info.threads = ths;
  info.mode = static_cast<PowerMode>(cls);
  auto provider = onnxruntime::make_unique<ARMExecutionProvider>(info);
  auto alloc = provider->GetAllocator(0, OrtMemTypeDefault);

  size_t size_a = tra? lda * n : lda * m;

  auto da = static_cast<float *>(alloc->Alloc(size_a * sizeof(float)));
  auto dx = static_cast<float *>(alloc->Alloc(n * sizeof(float)));
  auto dbias = static_cast<float *>(alloc->Alloc(m * sizeof(float)));
  auto dy = static_cast<float *>(alloc->Alloc(m * sizeof(float)));
  auto dy_basic = static_cast<float *>(alloc->Alloc(m * sizeof(float)));

  fill_data_rand(da, -1.f, 1.f, size_a);
  fill_data_rand(dx, -1.f, 1.f, n);
  fill_data_rand(dbias, -1.f, 1.f, m);
  fill_data_rand(dy, -1.f, 1.f, m);
//  fill_data_const(da, 1.f, size_a);
//  fill_data_const(dx, 1.f, n);
//  fill_data_const(dbias, 1.f, m);
//  fill_data_const(dy, 1.f, m);
  memcpy(dy_basic, dy, m * sizeof(float));

  if (check_result) {
    basic_gemv(tra, m, n, alpha, da, lda, dx, 1, beta, dy_basic, 1, dbias, has_bias, has_relu);
  }

  Timer t0;
  //! compute
  double ops = 2.0 * m * n;
  /// warmup
  for (int j = 0; j < warmup_iter; ++j) {
    arm::funcs::Sgemv(tra, m, n, alpha, da, lda, dx, 1, beta, dy, 1, dbias, has_bias, has_relu, provider.get());
  }

  t0.Reset();
  for (int i = 0; i < repeats; ++i) {
    t0.Start();
    arm::funcs::Sgemv(tra, m, n, alpha, da, lda, dx, 1, beta, dy, 1, dbias, has_bias, has_relu, provider.get());
    t0.Stop();
  }
  std::cout << ", GOPS: " << ops * 1e-9f << " GOPS" \
            << ", avg time: " << t0.LapTimes().Avg() << "ms"
            << ", min time: " << t0.LapTimes().Min() << "ms"
            << ", mean GOPs: " << ops * 1e-6f / t0.LapTimes().Avg() << "GOPS"
            << ", max GOPs: " << ops * 1e-6f / t0.LapTimes().Min() << "GOPs\n";

  if (check_result) {
    double max_ratio = 0;
    double max_diff = 0;
    /// fp32 result
    check_precision(dy_basic, dy, max_ratio, max_diff, m);
    std::cout << "compare result, max diff: " << max_diff
              << ", max ratio: " << max_ratio << std::endl;
    if (std::abs(max_ratio) > 1e-4f && std::abs(max_diff) > 5e-5f) {
      auto data_diff = static_cast<float *>(alloc->Alloc(m * sizeof(float)));
      compute_data_diff(dy_basic, dy, data_diff, m);
      std::cout << "basic result: \n";
      print_data(dy_basic, m, m);
      std::cout << "arm result: \n";
      print_data(dy, m, m);
      std::cout << "diff result: \n";
      print_data(data_diff, m, m);
      return false;
    }
  }
  return true;
}

TEST(TestARMSgemv, Sgemv) {
  if (precision_test) {
    std::cout << "run basic sgemv test";
    for (auto &m : {1, 3, 8, 21, 32, 397}) {
      for (auto &n : {1, 3, 8, 17, 59, 234}) {
        for (auto &tra : {true, false}) {
          for (auto& lda_inc : {0, 10}) {
            for (auto& alpha : {1.f, 0.5f}) {
              for (auto &beta : {0.f, 0.5f}) {
                for (auto &has_bias : {false, true}) {
                  for (auto &has_relu : {false, true}) {
                    for (auto &th : {1, 2, 4}) {
                      int lda = tra ? m : n;
                      lda += lda_inc;
                      auto flag = TestSgemv(tra, m, n, alpha, lda, beta,
                                             has_bias, has_relu, 0, th, 0);
                      if (flag) {
                        std::cout << "test m = " << m << ", n=" << n
                                  << ", bias: " << (has_bias ? "true" : "false")
                                  << ", relu: " << (has_relu ? "true" : "false")
                                  << ", trans A: " << (tra ? "true" : "false")
                                  << ", threads: " << th << " passed\n";
                      } else {
                        std::cout << "test m = " << m << ", n=" << n
                                  << ", bias: " << (has_bias ? "true" : "false")
                                  << ", relu: " << (has_relu ? "true" : "false")
                                  << ", trans A: " << (tra ? "true" : "false")
                                  << ", threads: " << th << " failed\n";
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

TEST(TestARMSgemv, Sgemv_performance) {
  if (performance_test) {
    int M = 512;
    int N = 512;
    for (auto& trans : {false, true}) {
      for (auto& th : {1, 2, 4}) {
        TestSgemv(trans, M, N, 1.f, M, 0.f,
                   false, false, 0, th, 50, 100, false);
      }
    }
  }
}

//TEST(TestSgemvCustom, Sgemv_custom) {
//#ifdef LITE_WITH_ARM
//  paddle::lite::DeviceInfo::Init();
//#endif
//  auto flag = test_sgemv(FLAGS_traA,
//                         FLAGS_M,
//                         FLAGS_K,
//                         FLAGS_flag_bias,
//                         FLAGS_flag_relu,
//                         FLAGS_cluster,
//                         FLAGS_threads);
//  if (!flag) {
//    LOG(FATAL) << "test m = " << FLAGS_M << ", k=" << FLAGS_K
//               << ", trans A: " << FLAGS_traA << ", bias: " << FLAGS_flag_bias
//               << ", relu: " << FLAGS_flag_relu << " failed!!";
//  }
//  LOG(INFO) << "test m = " << FLAGS_M << ", k=" << FLAGS_K
//            << ", trans A: " << FLAGS_traA << ", bias: " << FLAGS_flag_bias
//            << ", relu: " << FLAGS_flag_relu << " passed!!";
//}

}  //  test
}  //  onnxruntime