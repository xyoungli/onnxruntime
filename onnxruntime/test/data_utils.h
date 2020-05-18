#pragma once

#include <random>
template <typename Dtype>
inline void fill_data_const(Dtype* dio, Dtype value, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    dio[i] = value;
  }
}

template <typename Dtype>
inline void fill_data_rand(Dtype* dio, Dtype vstart, Dtype vend, size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 1.f);
  for (size_t i = 0; i < size; ++i) {
    dio[i] = static_cast<Dtype>(vstart + (vend - vstart) * dis(gen));
  }
}

template <typename Dtype>
inline void check_precision(const Dtype* in1, const Dtype* in2, double& max_ratio, double& max_diff, size_t size) {
  const double eps = 1e-6f;
  max_diff = fabs(in1[0] - in2[0]);
  max_ratio = fabs(max_diff) / (std::abs(in1[0]) + eps);
  for (size_t i = 1; i < size; ++i) {
    double diff = fabs(in1[i] - in2[i]);
    double ratio = fabs(diff) / (std::abs(in1[i]) + eps);
    if (max_ratio < ratio) {
      max_diff = diff;
      max_ratio = ratio;
    }
  }
}

template <typename Dtype>
inline void compute_data_diff(const Dtype* in1, const Dtype* in2, Dtype* out, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    out[i] = in1[i] - in2[i];
  }
}

template <typename Dtype>
inline void print_data(const Dtype* in, size_t size, size_t stride);

template <>
inline void print_data(const float* in, size_t size, size_t stride) {
  for (size_t i = 0; i < size; ++i) {
    printf("%0.6f ", in[i]);
    if ((i + 1) % stride == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

template <>
inline void print_data(const int* in, size_t size, size_t stride) {
  for (size_t i = 0; i < size; ++i) {
    printf("%d ", in[i]);
    if ((i + 1) % stride == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

template <>
inline void print_data(const int8_t* in, size_t size, size_t stride) {
  for (size_t i = 0; i < size; ++i) {
    printf("%d ", in[i]);
    if ((i + 1) % stride == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

bool CheckFp32(const float* basic, const float* result, size_t size, size_t ldc) {
  double max_ratio = 0;
  double max_diff = 0;
  check_precision(basic, result, max_ratio, max_diff, size);
  std::cout << "compare fp32 result: max diff: " << max_diff << ", max ratio: " << max_ratio << std::endl;
  if (std::abs(max_ratio) > 1e-4f && std::abs(max_diff) > 5e-5f) {
    auto data_diff = static_cast<float *>(malloc(size * sizeof(float)));
    compute_data_diff(basic, result, data_diff, size);
    std::cout << "basic result fp32: \n";
    print_data(basic, size, ldc);
    std::cout << "arm result fp32: \n";
    print_data(result, size, ldc);
    std::cout << "diff result: \n";
    print_data(data_diff, size, ldc);
    free(data_diff);
    return false;
  }
  return true;
}

bool CheckInt8(const int8_t* basic, const int8_t* result, size_t size, size_t ldc, float thresh, int max_num) {
  double max_ratio = 0;
  double max_diff = 0;
  check_precision(basic, result, max_ratio, max_diff, size);
  std::cout << "compare int8 result: max diff: " << max_diff << ", max ratio: " << max_ratio << std::endl;
  if (std::abs(max_ratio) > 1e-4f ) {
    auto data_diff = static_cast<int8_t*>(malloc(size * sizeof(int8_t)));
    compute_data_diff(basic, result, data_diff, size);
    float count = 0;
    bool check = true;
    for (int i = 0; i < size; ++i) {
      if (abs(data_diff[i]) > 1) {
        check = false;
        break;
      }
      if (data_diff[i] != 0) {
        count += 1;
      }
    }
    check = check && count < std::max(max_num, static_cast<int>(thresh * size));
    if (!check) {
      std::cout << "int8 basic result\n";
      print_data(basic, size, ldc);
      std::cout << "int8 arm result\n";
      print_data(result, size, ldc);
      std::cout << "int8 diff tensor\n";
      print_data(data_diff, size, ldc);
      return false;
    }
  }
  return true;
}