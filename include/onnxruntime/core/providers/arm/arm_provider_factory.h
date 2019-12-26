// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param use_arena zero: false. non-zero: true.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_ARM, _In_ OrtSessionOptions* options, int use_arena,
        PowerMode mode, int threads)
ORT_ALL_ARGS_NONNULL;

#ifdef __cplusplus
}
#endif
