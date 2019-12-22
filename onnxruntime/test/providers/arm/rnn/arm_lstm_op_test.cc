// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <iterator>
#include <vector>

#include "test/providers/provider_test_utils.h"
#include "core/providers/arm/arm_execution_provider.h"
#include "test/timer.h"

using namespace std;
namespace onnxruntime {
namespace test {

// copy the contents of the container to the end so the original values are duplicated
template <typename T>
T DuplicateContainer(const T& container) {
  T doubled;
  doubled.reserve(container.size() * 2);  // need to avoid reallocation when inserting
  std::copy(container.cbegin(), container.cend(), std::back_inserter(doubled));
  std::copy(container.cbegin(), container.cend(), std::back_inserter(doubled));
  return doubled;
}

static void RunLstmTest(const std::vector<float>& X_data,
                        const std::vector<float>& W_data,
                        const std::vector<float>& R_data,
                        const std::vector<float>& Y_data,
                        const std::vector<float>& Y_h_data,
                        const std::vector<float>& Y_c_data,
                        int64_t input_size,
                        int batch_size,
                        int64_t hidden_size,
                        int64_t seq_length,
                        const std::vector<float>* B_data = nullptr,
                        const std::vector<float>* P_data = nullptr,
                        const std::vector<float>* initial_h_data = nullptr,
                        const std::vector<float>* initial_c_data = nullptr,
                        const std::vector<int>* sequence_lengths = nullptr,
                        const std::string& direction = "forward",
                        float clip = 9999.f,
                        bool output_sequence = true,
                        bool input_forget = false,
                        // copy the following vectors as we may modify them
                        std::vector<string> activations = {},
                        std::vector<float> activation_alphas = {},
                        std::vector<float> activation_betas = {},
                        bool hasClip = true,
                        int warmup_iter=0,
                        int repeats=1,
                        int power_mode=0,
                        int threads=1) {
  OpTester test("LSTM");

  int num_directions = (direction == "bidirectional") ? 2 : 1;

  if (activations.empty()) {
    activations = {"sigmoid", "tanh", "tanh"};
  }

  if (num_directions == 2 && activations.size() == 3) {
    activations = DuplicateContainer(activations);
  }

  test.AddAttribute<std::vector<string>>("activations", activations);
  if (!activation_alphas.empty())
    test.AddAttribute<std::vector<float>>("activation_alpha", activation_alphas);
  if (!activation_betas.empty())
    test.AddAttribute<std::vector<float>>("activation_beta", activation_betas);

  test.AddAttribute("direction", direction);
  test.AddAttribute("hidden_size", hidden_size);
  // test.AddAttribute<int64_t>("output_sequence", output_sequence);
  test.AddAttribute<int64_t>("input_forget", input_forget);
  if (hasClip) {
    test.AddAttribute<float>("clip", clip);
  }

  std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
  std::vector<int64_t> W_dims = {num_directions, 4 * hidden_size, input_size};
  std::vector<int64_t> R_dims = {num_directions, 4 * hidden_size, hidden_size};

  test.AddInput<float>("X", X_dims, X_data);
  test.AddInput<float>("W", W_dims, W_data);
  test.AddInput<float>("R", R_dims, R_data);

  if (B_data) {
    std::vector<int64_t> B_dims = {num_directions, 8 * hidden_size};
    test.AddInput<float>("B", B_dims, *B_data);
  } else {
    test.AddMissingOptionalInput<float>();
  }

  if (sequence_lengths) {
    std::vector<int64_t> sequence_lens_dims{batch_size};
    test.AddInput<int>("sequence_lens", sequence_lens_dims, *sequence_lengths);
  } else {
    test.AddMissingOptionalInput<int>();
  }

  if (initial_h_data && !initial_h_data->empty()) {
    std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
    test.AddInput<float>("initial_h", initial_h_dims, *initial_h_data);
  } else {
    test.AddMissingOptionalInput<float>();
  }

  if (initial_c_data && !initial_c_data->empty()) {
    std::vector<int64_t> initial_c_dims = {num_directions, batch_size, hidden_size};
    test.AddInput<float>("initial_c", initial_c_dims, *initial_c_data);
  } else {
    test.AddMissingOptionalInput<float>();
  }

  if (P_data && !P_data->empty()) {
    std::vector<int64_t> P_dims = {num_directions, 3 * hidden_size};
    test.AddInput<float>("P", P_dims, *P_data);
  } else {
    test.AddMissingOptionalInput<float>();
  }

  if (output_sequence != 0 && !Y_data.empty()) {
    std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
    test.AddOutput<float>("Y", Y_dims, Y_data);
  } else {
    // add placeholder so node counts match as Y_h will always be the second Y_data,
    // so Y must exist as the first Y_data
    test.AddMissingOptionalOutput<float>();
  }

  if (!Y_h_data.empty()) {
    std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
    test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);
  } else {
    test.AddMissingOptionalOutput<float>();
  }

  if (!Y_c_data.empty()) {
    std::vector<int64_t> Y_c_dims{num_directions, batch_size, hidden_size};
    test.AddOutput<float>("Y_c", Y_c_dims, Y_c_data);
  } else {
    test.AddMissingOptionalOutput<float>();
  }

  ARMExecutionProviderInfo info;
  info.threads = threads;
  info.mode = static_cast<PowerMode>(power_mode);
  auto arm_provider = onnxruntime::make_unique<ARMExecutionProvider>(info);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(std::move(arm_provider));

  Timer t0;
  test.SetNumRunCalls(repeats);
  t0.Start();
  test.Run(OpTester::ExpectResult::kExpectSuccess,
          "",
          {kCpuExecutionProvider},
          nullptr,
          &providers,
          ExecutionMode::ORT_SEQUENTIAL);
  t0.Stop();
  std::cout << "LSTM, power_mode: " << power_mode << ", threads: " << threads
            << ", repeats: " << repeats
            << ", avg time: " << t0.LapTimes().Avg() / repeats << "ms\n";
//            << ", min time: " << t0.LapTimes().Min() << "ms\n";
}

void SimpleWeightsNoBiasTwoRows(const std::string& direction,
                                const std::vector<float>& Y_data,
                                const std::vector<float>& Y_h_data,
                                const std::vector<float>& Y_c_data,
                                const std::vector<int>* seq_lengths = nullptr) {
  int64_t seq_length = 2;
  int batch_size = 2;
  int64_t input_size = 1;
  int64_t hidden_size = 3;

  int num_directions = direction == "bidirectional" ? 2 : 1;

  std::vector<float> X_data{1.f, 2.f, 10.f, 11.f};

  std::vector<float> W_data{
      0.1f, 0.2f, 0.3f, 0.4f,
      1.f, 2.f, 3.f, 4.f,
      10.f, 11.f, 12.f, 13.f};

  std::vector<float> R_data(num_directions * 4 * hidden_size * hidden_size, 0.1f);

  // duplicate for bidirectional
  if (num_directions == 2) {
    W_data = DuplicateContainer(W_data);
  }

  RunLstmTest(X_data, W_data, R_data, Y_data, Y_h_data, Y_c_data,
              input_size, batch_size, hidden_size, seq_length,
              nullptr, nullptr, nullptr, nullptr, seq_lengths, direction);

  // need at least one output, so we need Y_h or Y_c to be requested (non-empty output to compare against) in order
  // to test Y not being returned (output_sequence == false)
//  if (!Y_h_data.empty() || !Y_c_data.empty()) {
//    RunLstmTest(X_data, W_data, R_data, Y_data, Y_h_data, Y_c_data,
//                input_size, batch_size, hidden_size, seq_length,
//                nullptr, nullptr, nullptr, nullptr, seq_lengths, direction, 999.f, /* output_sequence*/ false);
//  }
}

TEST(LSTMTest, ForwardSimpleWeightsNoBiasTwoRows) {
  std::vector<float> Y_data{
      0.28828835f, 0.36581863f, 0.45679406f,
      0.34526032f, 0.47220859f, 0.55850911f,

      0.84196719f, 0.89402526f, 0.91073048f,
      0.85882828f, 0.90703777f, 0.92382453f};

  std::vector<float> Y_h_data{
      0.84196719f, 0.89402526f, 0.91073048f,
      0.85882828f, 0.90703777f, 0.92382453f};

  std::vector<float> Y_c_data{
      1.27731147f, 1.44181041f, 1.53179041f,
      1.3249796f, 1.51063104f, 1.61451544f};

  SimpleWeightsNoBiasTwoRows("forward", Y_data, Y_h_data, Y_c_data);

  // test Y_h and Y_c being optional
//  SimpleWeightsNoBiasTwoRows("forward", Y_data, {}, {});
}
#if 0
TEST(LSTMTest, ReverseSimpleWeightsNoBiasTwoRows) {
  std::vector<float> Y_data{
      0.55391603f, 0.69201493f, 0.82696019f,
      0.64046413f, 0.82303363f, 0.91610711f,

      0.61249432f, 0.70678632f, 0.74094619f,
      0.62759886f, 0.71640738f, 0.74624585f};

  std::vector<float> Y_h_data{
      0.55391603f, 0.69201493f, 0.82696019f,
      0.64046413f, 0.82303363f, 0.91610711f};

  std::vector<float> Y_c_data{
      1.27850552f, 1.46799496f, 1.57641257f,
      1.34960834f, 1.54772296f, 1.65633056f};

  SimpleWeightsNoBiasTwoRows("reverse", Y_data, Y_h_data, Y_c_data);
}

TEST(LSTMTest, BidirectionalSimpleWeightsNoBiasTwoRows) {
  std::vector<float> Y_data{
      0.28828835f, 0.36581863f, 0.45679406f,
      0.34526032f, 0.47220859f, 0.55850911f,

      0.55391603f, 0.69201493f, 0.82696019f,
      0.64046413f, 0.82303363f, 0.91610711f,

      0.84196719f, 0.89402526f, 0.91073048f,
      0.85882828f, 0.90703777f, 0.92382453f,

      0.61249432f, 0.70678632f, 0.74094619f,
      0.62759886f, 0.71640738f, 0.74624585f};

  std::vector<float> Y_h_data{
      // we did the forward processing of X_data[1] last
      0.84196719f, 0.89402526f, 0.91073048f,
      0.85882828f, 0.90703777f, 0.92382453f,

      // and the reverse processing of X_data[0] last as the X_data order was reversed
      0.55391603f, 0.69201493f, 0.82696019f,
      0.64046413f, 0.82303363f, 0.91610711f};

  std::vector<float> Y_c_data{
      1.27731147f, 1.44181041f, 1.53179041f,
      1.3249796f, 1.51063104f, 1.61451544f,

      1.27850552f, 1.46799496f, 1.57641257f,
      1.34960834f, 1.54772296f, 1.65633056f};

  // cudnn don't support customized activation
  SimpleWeightsNoBiasTwoRows("bidirectional", Y_data, Y_h_data, Y_c_data);
}

TEST(LSTMTest, MixedSequenceLengths) {
  // we don't have numpy output for this, but by testing twice and swapping which batch is smaller
  // we can largely verify the behaviour by comparing to ForwardSimpleWeightsNoBiasTwoRows output.
  std::vector<int> seq_lengths{1, 2};

  std::vector<float> Y_data{
      0.28828835f, 0.36581863f, 0.45679406f,
      0.34526032f, 0.47220859f, 0.55850911f,

      0.f, 0.f, 0.f,
      0.85882828f, 0.90703777f, 0.92382453f};

  std::vector<float> Y_h_data{
      0.28828835f, 0.36581863f, 0.45679406f,
      0.85882828f, 0.90703777f, 0.92382453f};

  std::vector<float> Y_c_data{
      0.52497941f, 0.54983425f, 0.5744428f,  // see intermediate output from ForwardSimpleWeightsNoBiasTwoRows
      1.3249796f, 1.51063104f, 1.61451544f};

  // Not able to mask on Y_c for CUDA using cudnn lib
  SimpleWeightsNoBiasTwoRows("forward", Y_data, Y_h_data, Y_c_data, &seq_lengths);

  // swap which one is short
  seq_lengths = {2, 1};

  Y_data = {
      0.28828835f, 0.36581863f, 0.45679406f,
      0.34526032f, 0.47220859f, 0.55850911f,

      0.84196719f, 0.89402526f, 0.91073048f,
      0.f, 0.f, 0.f};

  Y_h_data = {
      0.84196719f, 0.89402526f, 0.91073048f,
      0.34526032f, 0.47220859f, 0.55850911f};

  Y_c_data = {
      1.27731147f, 1.44181041f, 1.53179041f,
      0.54983425f, 0.59868795f, 0.64565659f};

  SimpleWeightsNoBiasTwoRows("forward", Y_data, Y_h_data, Y_c_data, &seq_lengths);
}

TEST(LSTMTest, MixedSequenceLengthsReverse) {
  // we don't have numpy output for this, but by testing twice and swapping which batch is smaller
  // we can largely verify the behaviour by comparing to ReverseSimpleWeightsNoBiasTwoRows output.
  std::vector<int> seq_lengths{1, 2};

  std::vector<float> Y_data{
      0.28828844f, 0.36581877f, 0.45679423f,
      0.64046413f, 0.82303363f, 0.91610711f,

      0.f, 0.f, 0.f,
      0.62759886f, 0.71640738f, 0.74624585f};

  std::vector<float> Y_h_data{
      0.28828844f, 0.36581877f, 0.45679423f,
      0.64046413f, 0.82303363f, 0.91610711f};

  std::vector<float> Y_c_data{
      0.52497941f, 0.54983425f, 0.5744428f,
      1.34960834f, 1.54772296f, 1.65633056f};

  SimpleWeightsNoBiasTwoRows("reverse", Y_data, Y_h_data, Y_c_data, &seq_lengths);

  // swap which one is short
  seq_lengths = {2, 1};

  Y_data = {
      0.55391603f, 0.69201493f, 0.82696019f,
      0.34526044f, 0.47220877f, 0.55850935f,

      0.61249432f, 0.70678632f, 0.74094619f,
      0.f, 0.f, 0.f};

  Y_h_data = {
      0.55391603f, 0.69201493f, 0.82696019f,
      0.34526044f, 0.47220877f, 0.55850935f};

  Y_c_data = {
      1.27850552f, 1.46799496f, 1.57641257f,
      0.54983425f, 0.59868795f, 0.64565659f};

  SimpleWeightsNoBiasTwoRows("reverse", Y_data, Y_h_data, Y_c_data, &seq_lengths);
}

// test path in LSTM model where batch_parallel_ is false and there are multiple steps (seq_length > 1)
TEST(LSTMTest, BatchParallelFalseSeqLengthGreaterThanOne) {
  int64_t seq_length = 2;
  int batch_size = 1;
  int64_t input_size = 1;
  int64_t hidden_size = 2;

  int num_directions = 1;

  std::vector<float> X_data{1.f, 2.f};

  std::vector<float> W_data{
      0.1f, 0.2f, 0.3f, 0.4f,
      1.f, 2.f, 3.f, 4.f};

  std::vector<float> R_data(num_directions * 4 * hidden_size * hidden_size, 0.1f);

  std::vector<float> Y_data{
      0.27546653f, 0.29941525f,
      0.50903179f, 0.57476457f};

  std::vector<float> Y_c_data{
      1.02721067f, 1.15254318f};

  RunLstmTest(X_data, W_data, R_data, Y_data, {}, Y_c_data,
              input_size, batch_size, hidden_size, seq_length);
}

// make sure GateComputations works correctly if batch_parallel_ is true due to large batch size
static void LargeBatchWithClip(const std::vector<float>& Y_h_data, float clip = 9999.0) {
  int64_t seq_length = 2;
  int batch_size = 32;
  int64_t input_size = 1;
  int64_t hidden_size = 3;

  const std::string direction = "forward";
  int num_directions = 1;

  std::vector<float> X_data;

  // generate input of 64 values
  float i = 0.f, increment = 1.f;
  std::generate_n(std::back_inserter(X_data), batch_size * seq_length, [&]() { return i += increment; });

  std::vector<float> W_data{0.1f, 0.2f, 0.3f, 0.4f,
                            1.f, 2.f, 3.f, 4.f,
                            10.f, 11.f, 12.f, 13.f};

  std::vector<float> R_data(num_directions * 4 * hidden_size * hidden_size, 0.1f);

  RunLstmTest(X_data, W_data, R_data, {}, Y_h_data, {},
              input_size, batch_size, hidden_size, seq_length,
              nullptr, nullptr, nullptr, nullptr, nullptr, direction, clip);
}

TEST(LSTMTest, LargeBatchNoClipping) {
  std::vector<float> Y_h_data = {
      0.90387899f, 0.9135572f, 0.91772245f,
      0.90897038f, 0.92132433f, 0.92825467f,
      0.91365823f, 0.92815113f, 0.93676105f,
      0.91799162f, 0.93406357f, 0.94344562f,
      0.92199681f, 0.93912057f, 0.94859476f,
      0.92569357f, 0.94340185f, 0.95250664f,
      0.92909964f, 0.94699686f, 0.95545127f,
      0.93223207f, 0.94999634f, 0.95765468f,
      0.93510761f, 0.9524867f, 0.95929726f,
      0.93774272f, 0.9545467f, 0.96051891f,
      0.9401536f, 0.95624603f, 0.96142619f,
      0.94235605f, 0.95764499f, 0.96209939f,
      0.94436539f, 0.95879495f, 0.96259862f,
      0.94619635f, 0.95973921f, 0.96296872f,
      0.94786299f, 0.96051397f, 0.96324302f,
      0.94937864f, 0.96114929f, 0.96344629f,
      0.95075587f, 0.96167006f, 0.96359692f,
      0.95200645f, 0.96209679f, 0.96370852f,
      0.95314133f, 0.9624464f, 0.9637912f,
      0.95417069f, 0.96273278f, 0.96385246f,
      0.95510395f, 0.96296733f, 0.96389785f,
      0.95594975f, 0.96315942f, 0.96393147f,
      0.95671607f, 0.96331673f, 0.96395638f,
      0.9574102f, 0.96344554f, 0.96397483f,
      0.9580388f, 0.96355102f, 0.9639885f,
      0.95860795f, 0.96363739f, 0.96399863f,
      0.95912322f, 0.96370811f, 0.96400613f,
      0.95958963f, 0.96376601f, 0.96401169f,
      0.96001179f, 0.96381342f, 0.96401581f,
      0.96039386f, 0.96385224f, 0.96401886f,
      0.96073964f, 0.96388402f, 0.96402112f,
      0.96105254f, 0.96391004f, 0.96402279f};

  LargeBatchWithClip(Y_h_data);
}

// make sure GateComputations with clipping works correctly if batch_parallel_ is true due to large batch size
TEST(LSTMTest, LargeBatchWithClip) {
  std::vector<float> Y_h_data = {
      0.88572926f, 0.89251395f, 0.89655037f,
      0.89074291f, 0.90035688f, 0.90727429f,
      0.89535827f, 0.90727429f, 0.91596163f,
      0.89963124f, 0.91328279f, 0.9228067f,
      0.90358195f, 0.91843507f, 0.92809163f,
      0.90723279f, 0.9228067f, 0.93211437f,
      0.91038955f, 0.92648469f, 0.93514718f,
      0.91328279f, 0.92955856f, 0.93741938f,
      0.91596163f, 0.93211437f, 0.9391149f,
      0.91843507f, 0.93423112f, 0.94037686f,
      0.92071318f, 0.9359791f, 0.94131462f,
      0.9228067f, 0.93741938f, 0.94201073f,
      0.92472679f, 0.9386042f, 0.94252713f,
      0.92648469f, 0.9395777f, 0.94266769f,
      0.92809163f, 0.94037686f, 0.94266769f,
      0.92955856f, 0.94103248f, 0.94266769f,
      0.93089609f, 0.94157007f, 0.94266769f,
      0.93211437f, 0.94201073f, 0.94266769f,
      0.93322302f, 0.94237184f, 0.94266769f,
      0.93423112f, 0.94266769f, 0.94266769f,
      0.93514718f, 0.94266769f, 0.94266769f,
      0.9359791f, 0.94266769f, 0.94266769f,
      0.93673424f, 0.94266769f, 0.94266769f,
      0.93741938f, 0.94266769f, 0.94266769f,
      0.93804079f, 0.94266769f, 0.94266769f,
      0.9386042f, 0.94266769f, 0.94266769f,
      0.9391149f, 0.94266769f, 0.94266769f,
      0.9395777f, 0.94266769f, 0.94266769f,
      0.93999702f, 0.94266769f, 0.94266769f,
      0.94037686f, 0.94266769f, 0.94266769f,
      0.94072091f, 0.94266769f, 0.94266769f,
      0.94103248f, 0.94266769f, 0.94266769f};

  LargeBatchWithClip(Y_h_data, 4.f);
}
#endif
}  // namespace test
}  // namespace onnxruntime
