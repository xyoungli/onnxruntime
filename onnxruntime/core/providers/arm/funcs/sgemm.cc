#include "sgemm.h"
#include "sgemm_packed.h"
#include "sgemv.h"

namespace onnxruntime {
namespace arm {
namespace funcs {

void Sgemm(bool transA,
           bool transB,
           int M,
           int N,
           int K,
           float alpha,
           const float* A,
           int lda,
           const float* B,
           int ldb,
           float beta,
           float* C,
           int ldc,
           const float* bias,
           bool is_bias,
           bool is_relu,
           ARMExecutionProvider* ctx) {
  auto alloc_ptr = ctx->GetAllocator(0, OrtMemTypeDefault);
  if (M == 1) {
    const float* x_ptr = nullptr;
    float* ptr_new = nullptr;
    if (transA && lda != M && K > 1) {
      ptr_new = static_cast<float*>(
              alloc_ptr->Alloc(K * sizeof(float)));
      for (int i = 0; i < K; i++) {
        ptr_new[i] = A[i * lda];
      }
      x_ptr = ptr_new;
    } else {
      x_ptr = A;
    }
    const float* bias_ptr = nullptr;
    float* bias_new = nullptr;
    if (N > 1 && is_bias) {
      bias_new = static_cast<float*>(
              alloc_ptr->Alloc(N * sizeof(float)));
      for (int i = 0; i < N; i++) {
        bias_new[i] = bias[0];
      }
      bias_ptr = bias_new;
    } else {
      bias_ptr = bias;
    }

    bool trans = !transB;
    int M_new = N;
    int N_new = K;
    auto A_new = B;
    int lda_new = ldb;
    Sgemv(trans, M_new, N_new, alpha, A_new, lda_new, x_ptr, 1, beta, C, 1, bias_ptr, is_bias, is_relu, ctx);
    if (ptr_new) {
      alloc_ptr->Free(ptr_new);
    }
    if (bias_new) {
      alloc_ptr->Free(bias_new);
    }
    return;
  }
  if (N == 1) {
    int incx = 1;
    if (!transB && ldb > 1) {
      incx = ldb;
    }
    return Sgemv(transA, M, K, alpha, A, lda, B, incx, beta, C, ldc, bias, is_bias, is_relu, ctx);
  }
  int hblock = GetSgemmHblock(ctx, M);
  int m_roundup = hblock * ((M + hblock - 1) / hblock);

  auto packed_A_ptr = static_cast<float*>(
          alloc_ptr->Alloc(m_roundup * K * sizeof(float)));

  PrepackA(packed_A_ptr, A, alpha, lda, 0, M, 0, K, transA, ctx);

  SgemmPrepack(transB, M, N, K,
               packed_A_ptr, B, ldb,
               beta, C, ldc,
               bias, is_bias, is_relu, ctx);
  alloc_ptr->Free(packed_A_ptr);
}

}  // namespace funcs
}  // namespace arm
}  // namespace onnxruntime
