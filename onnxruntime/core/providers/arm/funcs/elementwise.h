#pragma once

namespace onnxruntime {
namespace arm {
namespace funcs {

template <typename T>
void ElementwiseAdd(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void ElementwiseAddReLU(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void ElementwiseAddBroadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void ElementwiseAddReLUBroadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void ElementwiseSub(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void ElementwiseSubReLU(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void ElementwiseSubBroadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void ElementwiseSubReLUBroadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void ElementwiseMul(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void ElementwiseMulReLU(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void ElementwiseMulBroadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void ElementwiseMulReLUBroadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void ElementwiseMac(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void ElementwiseMacReLU(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void ElementwiseMax(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void ElementwiseMaxReLU(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void ElementwiseMaxBroadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void ElementwiseMaxReLUBroadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void ElementwiseDiv(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void ElementwiseDivBroadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

template <typename T>
void ElementwiseDivReLU(const T* dinx, const T* diny, T* dout, int num);

template <typename T>
void ElementwiseDivReLUBroadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num);

}  // namespace funcs
}  // namespace arm
}  // namespace onnxruntime
