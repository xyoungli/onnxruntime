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

void test_model(const char* model_path, int batch_size, int warmup_iter, int repeats, int power_mode, int threads) {
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  // initialize session options if needed
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
  // session (we also need to include cuda_provider_factory.h above which defines it)
  // #include "cuda_provider_factory.h"
  // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);

#ifdef USE_ARM
  auto state = OrtSessionOptionsAppendExecutionProvider_ARM(session_options,
          1, static_cast<PowerMode>(power_mode), threads);
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

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  size_t num_output_nodes = session.GetOutputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<const char*> output_node_names(num_output_nodes);
  std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
  // Otherwise need vector<vector<>>
  std::vector<int64_t> output_node_dims;

  printf("Number of inputs = %zu, Number of outputs = %zu\n", num_input_nodes, num_output_nodes);

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
    input_node_dims = tensor_info.GetShape();
    printf(", num_dims=%zu ", input_node_dims.size());
    for (int j = 0; j < input_node_dims.size(); j++)
      printf(", dim %d=%jd", j, input_node_dims[j]);
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
    output_node_dims = tensor_info.GetShape();
    printf(", num_dims=%zu ", output_node_dims.size());
    for (int j = 0; j < output_node_dims.size(); j++)
      printf(", dim %d=%jd", j, output_node_dims[j]);
  }
  printf("\n");

  // Results should be...
  // Number of inputs = 1
  // Input 0 : name = data_0
  // Input 0 : type = 1
  // Input 0 : num_dims = 4
  // Input 0 : dim 0 = 1
  // Input 0 : dim 1 = 3
  // Input 0 : dim 2 = 224
  // Input 0 : dim 3 = 224

  //*************************************************************************
  // Similar operations to get output node information.
  // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
  // OrtSessionGetOutputTypeInfo() as shown above.

  //*************************************************************************
  // Score the model using sample data, and inspect values

  size_t input_tensor_size = 1;
  input_node_dims[0] = batch_size;
  for (auto& d : input_node_dims) {
    input_tensor_size *= d;
  }

  std::vector<float> input_tensor_values(input_tensor_size);

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());

  std::vector<Ort::Value> output_tensors;
  for (int i = 0; i < warmup_iter; ++i) {
    session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
            &input_tensor, 1, output_node_names.data(), 1);
  }
  onnxruntime::Timer t0;
  // score model & input tensor, get back output tensor
  for (int i = 0; i < repeats; ++i) {
    t0.Start();
    output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
            &input_tensor, 1, output_node_names.data(), 1);
    t0.Stop();
  }
  std::cout << "================== Speed Report ===================\n";
  std::cout << "Model: " << model_path
            << ", power_mode: " << power_mode
            << ", threads num " << threads
            << ", warmup: " << warmup_iter << ", repeats: " << repeats
            << ", avg time: " << t0.LapTimes().Avg()
            << " ms"
            << ", min time: " << t0.LapTimes().Min() << " ms"
            << ", max time: " << t0.LapTimes().Max() << " ms.\n";

  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  auto floatarr = output_tensors.front().GetTensorMutableData<float>();
  assert(fabs(floatarr[0] - 0.000045) < 1e-6);

  // score the model, and print scores for first 5 classes
  for (int i = 0; i < 5; i++)
    printf("Score for class [%d] =  %f\n", i, floatarr[i]);

  // Results of squeezenet should be as below...
  // Score for class[0] = 0.000045
  // Score for class[1] = 0.003846
  // Score for class[2] = 0.000125
  // Score for class[3] = 0.001180
  // Score for class[4] = 0.001317
  printf("Done!\n");
}

int main(int argc, char* argv[]) {
  std::cout << "usage:";
  std::cout << argv[0] << " <model path> <batch_size> <warmup_iter> <repeats> <power_mode> <threads>\n";
  std::cout << "   model path:     path to onnx model\n";
  std::cout << "   batch size:     batch size of input\n";
  std::cout << "   warmup_iter:    warm up iterations default to 1\n";
  std::cout << "   repeats:        repeat number for inference default to 10\n";
  std::cout << "   power_mode:     choose arm power mode, 0: threads not bind to specify cores, 1: big cores, 2: small cores, 3: all cores\n";
  std::cout << "   threads:        set openmp threads(omp threads)\n";
  if(argc < 2) {
    std::cout << "You should fill in the variable lite model at least.\n";
    return 0;
  }
  const char* model_path = argv[1];
  int batch = 1;
  if (argc > 2) {
    batch = atoi(argv[2]);
  }
  int warmup_iter = 5;
  if (argc > 3) {
    warmup_iter = atoi(argv[3]);
  }
  int repeats = 10;
  if (argc > 4) {
    repeats = atoi(argv[4]);
  }
  int power_mode = 0;
  if (argc > 5) {
    power_mode = atoi(argv[5]);
    if (power_mode < 0) {
      power_mode = 0;
    }
    if (power_mode > 3) {
      power_mode = 3;
    }
  }
  int threads = 1;
  if (argc > 6) {
    threads = atoi(argv[6]);
  }
  test_model(model_path, batch, warmup_iter, repeats, power_mode, threads);
  return 0;
}
