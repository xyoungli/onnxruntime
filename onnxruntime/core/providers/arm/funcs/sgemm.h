#pragma once

#include <cmath>
#include "core/providers/arm/arm_execution_provider.h"

namespace onnxruntime {
namespace arm {

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
           ARMExecutionProvider* ctx);

void PackA(float* out,
           const float* in,
           float alpha,
           int ldin,
           int M,
           int K,
           bool trans,
           ARMExecutionProvider* ctx);

void PackB(float* out,
           const float* in,
           int ldin,
           int K,
           int N,
           bool trans,
           ARMExecutionProvider* ctx);

}  // namespace arm
}  // namespace onnxruntime
