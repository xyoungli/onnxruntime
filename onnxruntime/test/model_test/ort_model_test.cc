#include <vector>
#include <iostream>

#include "onnxruntime_cxx_api.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#endif
#include "timer.h"
#include "data_utils.h"

void test_model(const char* model_path, std::vector<std::string>& providers,
                std::vector<std::vector<int64_t>>& input_shapes,
                int warmup_iter, int repeats, int power_mode, int threads,
                const char* optimized_model_path = nullptr) {
  (void)(providers);
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info
//  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  // initialize session options if needed
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
//  session_options.EnableProfiling("profile.txt");
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
  (void)(use_device);
#ifdef USE_CUDA
  if (use_device("cuda")) {
    printf("add provider: cuda\n");
    auto state = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, power_mode);
    (void)(state);
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
    if (input_shapes.size() != num_input_nodes) {
      std::cerr << "number of input shapes must equal to number of input nodes, got:"
                << input_shapes.size() <<  ", expect: " << num_input_nodes << std::endl;
      return;
    }
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
      printf(", dim %d=%ld", j, dims[j]);
      if (dims[j] < 0) {
        dims[j] = 1;
        printf(" is < 0, set to 1");
      }
    }
    if (!input_shapes.empty()) {
      if (input_shapes[i].size() != dims.size()) {
        std::cerr << "input shape size must equal to tensor's dims, got: "
                  << input_shapes[i].size() << ", expect: " << dims.size() << std::endl;
        return;
      }
      dims = input_shapes[i];
      printf(", reset size to user input: ");
      for (int j = 0; j < dims.size(); j++) {
        printf(", dim %d=%ld", j, dims[j]);
      }
    }
    printf("\n");
    input_node_dims.push_back(dims);
    // create tensor and set data
    size_t size = 1;
    for (int k = 0; k < dims.size(); ++k) {
      size *= dims[k];
    }
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator, dims.data(), dims.size());
      if (!input_tensor.IsTensor()) {
        std::cerr << "input must be a tensor\n";
        return;
      }
      fill_data_const(input_tensor.GetTensorMutableData<float>(), 1.f, size);
      input_tensors.push_back(std::move(input_tensor));
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      Ort::Value input_tensor = Ort::Value::CreateTensor<int32_t>(allocator, dims.data(), dims.size());
      if (!input_tensor.IsTensor()) {
        std::cerr << "input must be a tensor\n";
        return;
      }
      fill_data_const(input_tensor.GetTensorMutableData<int32_t>(), 2, size);
      input_tensors.push_back(std::move(input_tensor));
    } else {
      std::cerr << "unsupport input data type: " << type << std::endl;
      return;
    }

  }
  printf("\n");

//  auto ptr0 = input_tensors[0].GetTensorMutableData<int>();  // pos
//  auto ptr1 = input_tensors[1].GetTensorMutableData<int>();  // length
//  auto ptr2 = input_tensors[2].GetTensorMutableData<int>();  // input
//  auto ptr3 = input_tensors[3].GetTensorMutableData<int>();  // char
//
//  std::vector<int> input_ids = {20980,375,34,5593,1798,35,7179,17,3511,663,20982,7147,8420,35,20982,605,14,20981};
//  std::vector<int> char_ids = {5479,0,1669,0,4154,0,1954,2265,1441,0,4480,0,511,2313,110,0,4626,0,1714,0,5471,854,
//                               1581,2368,379,0,4480,0,4934,2644,3116,0,2043,0,5480,0};
//  std::vector<int> length = {1,1,1,2,1,1,2,1,1,1,2,2,1,1,2,1,1,1};
//  std::vector<int> pos_ids = {57,42,29,4,54,29,38,45,4,18,4,40,54,29,4,16,45,0};
//
//  std::memcpy(ptr0, pos_ids.data(), sizeof(int) * pos_ids.size());
//  std::memcpy(ptr1, length.data(), sizeof(int) * length.size());
//  std::memcpy(ptr2, input_ids.data(), sizeof(int) * input_ids.size());
//  std::memcpy(ptr3, char_ids.data(), sizeof(int) * char_ids.size());

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
      printf(", dim %d=%ld", j, dims[j]);
    }
    printf("\n");
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

  // actual output shape
  // iterate over all output nodes
  std::vector<int64_t> output_sizes(num_output_nodes);
  std::vector<ONNXTensorElementDataType> output_types(num_output_nodes);
  for (int i = 0; i < num_output_nodes; i++) {
    Ort::TypeInfo type_info = output_tensors[0].GetTypeInfo();
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    output_types[i] = type;
    printf("output name: %s, data type=%d ", output_node_names[i], type);
    auto dims = tensor_info.GetShape();
    printf(", num_dims=%zu ", dims.size());
    int64_t size = 1;
    for (int j = 0; j < dims.size(); j++) {
      printf(", dim %d=%ld", j, dims[j]);
      size *= dims[j];
    }
    printf("\n");
    output_sizes[i] = size;
  }
  printf("\n");

  // Get pointer to output tensor float values
  auto floatarr = output_tensors[0].GetTensorMutableData<float>();
  auto intarr = reinterpret_cast<int*>(floatarr);
  (void)(intarr);

  for (int i = 0; i < std::min(static_cast<int64_t>(100), output_sizes[0]); i++) {
    if (output_types[0] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      printf("output[%d] =  %f\n", i, floatarr[i]);
    } else if (output_types[0] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      printf("output[%d] =  %d\n", i, intarr[i]);
    } else {
      std::cerr << "unsupport input data type: " << output_types[0] << std::endl;
      return;
    }
  }
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

  int warmup_iter = 0;
  if (argc > 4) {
    warmup_iter = atoi(argv[4]);
    printf("warmup iters: %d\n", warmup_iter);
  }
  int repeats = 1;
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
