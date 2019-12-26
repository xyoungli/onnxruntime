// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//
#include <vector>
#include <iostream>

#include "onnxruntime_cxx_api.h"
#ifdef USE_ARM
#include "core/providers/arm/arm_provider_factory.h"
#endif
#include "test/timer.h"
#include "test/data_utils.h"

void test_model(const char* model_path, std::vector<std::string>& providers,
                std::vector<std::vector<int64_t>>& input_shapes,
                int warmup_iter, int repeats, int power_mode, int threads,
                const char* optimized_model_path = nullptr) {
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info
//  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "test");

  // initialize session options if needed
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.EnableProfiling("profile.txt");
  if (optimized_model_path) {
    session_options.SetOptimizedModelFilePath(optimized_model_path);
  }

  // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
  // session (we also need to include cuda_provider_factory.h above which defines it)
  // #include "cuda_provider_factory.h"
  // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);
  auto use_device = [&](const std::string& device_name) {
    if (providers.empty()) return false;
    for (int i = 0; i < providers.size(); ++i)
      if (device_name == providers[i]) return true;
    return false;
  };
#ifdef USE_ARM
  if (use_device("arm")) {
    printf("add provider: arm\n");
    auto state = OrtSessionOptionsAppendExecutionProvider_ARM(session_options,
                                                              1, static_cast<PowerMode>(power_mode), threads);
  }
#endif

  // Sets graph optimization level
  // Available levels are
  // ORT_DISABLE_ALL -> To disable all optimizations
  // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
  // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
  // ORT_ENABLE_ALL -> To Enable All possible opitmizations
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  //*************************************************************************
  // create session and load model into memory
  printf("Using Onnxruntime C++ API\n");
  Ort::Session session(env, model_path, session_options);

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;
  size_t num_input_nodes = session.GetInputCount();
  size_t num_output_nodes = session.GetOutputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<const char*> output_node_names(num_output_nodes);
  std::vector<std::vector<int64_t>> input_node_dims;
  std::vector<std::vector<int64_t>> output_node_dims;

  printf("Number of inputs = %zu, Number of outputs = %zu\n", num_input_nodes, num_output_nodes);

  if (!input_shapes.empty()) {
    assert(input_shapes.size() == num_input_nodes);
  }
  std::vector<Ort::Value> input_tensors;
  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name = session.GetInputName(i, allocator);
    printf("Input %d : name=%s ", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf(", type=%d ", type);

    // print input shapes/dims
    auto dims = tensor_info.GetShape();
    printf(", num_dims=%zu ", dims.size());
    for (int j = 0; j < dims.size(); j++) {
      printf(", dim %d=%jd", j, dims[j]);
      if (dims[j] < 0) {
        dims[j] = 1;
        printf(" is < 0, set to 1");
      }
    }
    if (!input_shapes.empty()) {
      assert(input_shapes[i].size() == dims.size());
      dims = input_shapes[i];
      printf(", reset size to user input: ");
      for (int j = 0; j < dims.size(); j++) {
        printf(", dim %d=%jd", j, dims[j]);
      }
    }
    printf("\n");
    input_node_dims.push_back(dims);
    // create tensor and set data
    size_t size = 1;
    for (int k = 0; k < dims.size(); ++k) {
      size *= dims[k];
    }
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator, dims.data(), dims.size());
    assert(input_tensor.IsTensor());
    fill_data_const(input_tensor.GetTensorMutableData<float>(), 1.f, size);
    input_tensors.push_back(std::move(input_tensor));
  }
  printf("\n");

  // iterate over all output nodes
  for (int i = 0; i < num_output_nodes; i++) {
    // print input node names
    char* output_name = session.GetOutputName(i, allocator);
    printf("Output %d : name=%s ", i, output_name);
    output_node_names[i] = output_name;

    // print output node types
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf(", type=%d ", type);

    // print input shapes/dims
    auto dims = tensor_info.GetShape();
    printf(", num_dims=%zu ", dims.size());
    for (int j = 0; j < dims.size(); j++) {
      printf(", dim %d=%jd", j, dims[j]);
    }
    output_node_dims.push_back(dims);
  }
  printf("\n");

  //*************************************************************************
  // run the model using sample data, and inspect values
  std::vector<Ort::Value> output_tensors;
  for (int i = 0; i < warmup_iter; ++i) {
    session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
            input_tensors.data(), num_input_nodes, output_node_names.data(), num_output_nodes);
  }
  onnxruntime::Timer t0;
  // score model & input tensor, get back output tensor
  for (int i = 0; i < repeats; ++i) {
    t0.Start();
    output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
            input_tensors.data(), num_input_nodes, output_node_names.data(), num_output_nodes);
    auto t = t0.Stop();
    printf("repeat number: %d, time: %0.6fms\n", i, t);
  }
  std::cout << "================== Speed Report ===================\n";
  std::cout << "Model: " << model_path
            << ", power_mode: " << power_mode
            << ", threads num " << threads
            << ", warmup: " << warmup_iter << ", repeats: " << repeats
            << ", avg time: " << t0.LapTimes().Avg() << " ms"
            << ", max time: " << t0.LapTimes().Max() << " ms"
            << ", min time: " << t0.LapTimes().Min() << " ms.\n";

  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  auto floatarr = output_tensors.front().GetTensorMutableData<float>();
  assert(fabs(floatarr[0] - 0.000045) < 1e-6);

  // actual output shape
  // iterate over all output nodes
  for (int i = 0; i < num_output_nodes; i++) {
    char* output_name = session.GetOutputName(i, allocator);
    printf("Actual Output %d : name=%s ", i, output_name);
    output_node_names[i] = output_name;
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf(", type=%d ", type);
    auto dims = tensor_info.GetShape();
    printf(", num_dims=%zu ", dims.size());
    for (int j = 0; j < dims.size(); j++)
      printf(", dim %d=%jd", j, dims[j]);
  }
  printf("\n");

  // score the model, and print scores for first 5 classes
  for (int i = 0; i < 5; i++)
    printf("output[%d] =  %f\n", i, floatarr[i]);
  printf("Done!\n");
}

int main(int argc, char* argv[]) {
  std::cout << "usage:";
  std::cout << argv[0] << " <model path> <providers> <input_size> <warmup_iter> <repeats> <power_mode> <threads>\n";
  std::cout << "   model path:     path to onnx model\n";
  std::cout << "   providers:      a list of providers, eg. 'cpu;arm'\n";
  std::cout << "   input size:     a list of input size, eg. '1, 3, 224, 224; 100, 100'\n";
  std::cout << "   warmup_iter:    warm up iterations default to 1\n";
  std::cout << "   repeats:        repeat number for inference default to 10\n";
  std::cout << "   power_mode:     choose arm power mode, 0: threads not bind to specify cores, 1: big cores, 2: small cores, 3: all cores\n";
  std::cout << "   threads:        set openmp threads(omp threads)\n";
  if(argc < 2) {
    std::cout << "You should fill in the variable lite model at least.\n";
    return 0;
  }
  const char* model_path = argv[1];

  auto split_string =
          [](const std::string& str_in) -> std::vector<std::string> {
            std::vector<std::string> str_out;
            std::string tmp_str = str_in;
            while (!tmp_str.empty()) {
              size_t next_offset = tmp_str.find(";");
              str_out.push_back(tmp_str.substr(0, next_offset));
              if (next_offset == std::string::npos) {
                break;
              } else {
                tmp_str = tmp_str.substr(next_offset + 1);
              }
            }
            return str_out;
          };

  auto get_shape = [](const std::string& str_shape) -> std::vector<int64_t> {
    std::vector<int64_t> shape;
    std::string tmp_str = str_shape;
    while (!tmp_str.empty()) {
      int dim = atoi(tmp_str.data());
      shape.push_back(dim);
      size_t next_offset = tmp_str.find(",");
      if (next_offset == std::string::npos) {
        break;
      } else {
        tmp_str = tmp_str.substr(next_offset + 1);
      }
    }
    return shape;
  };

  std::vector<std::string> providers = {"cpu"};
  if (argc > 2) {
    providers = split_string(argv[2]);
    for (int i = 0; i < providers.size(); ++i) {
      printf("providers: %s\n", providers[i].c_str());
    }
  }

  std::vector<std::vector<int64_t>> input_shapes;
  if (argc > 3) {
    printf("input shapes: %s\n", argv[3]);
    std::vector<std::string> str_input_shapes = split_string(argv[3]);
    for (int i = 0; i < str_input_shapes.size(); ++i) {
      printf("%d: input shape: %s\n", i, str_input_shapes[i].c_str());
      input_shapes.push_back(get_shape(str_input_shapes[i]));
    }
  }

  int warmup_iter = 5;
  if (argc > 4) {
    warmup_iter = atoi(argv[4]);
    printf("warmup iters: %d\n", warmup_iter);
  }
  int repeats = 10;
  if (argc > 5) {
    repeats = atoi(argv[5]);
    printf("repeats: %d\n", repeats);
  }
  int power_mode = 0;
  if (argc > 6) {
    power_mode = atoi(argv[6]);
    if (power_mode < 0) {
      power_mode = 0;
    }
    if (power_mode > 3) {
      power_mode = 3;
    }
    printf("power mode: %d\n", power_mode);
  }
  int threads = 1;
  if (argc > 7) {
    threads = atoi(argv[7]);
    printf("threads: %d\n", threads);
  }
  test_model(model_path, providers, input_shapes, warmup_iter, repeats, power_mode, threads, "optimized.onnx");
  return 0;
}
