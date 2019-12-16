#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#requires python3.6 or higher
python3 $DIR/tools/ci_build/build.py --parallel --use_openmp --config Release --build_dir $DIR/build/Mac --android --android_ndk_path /Users/lixiaoyang/android-ndk-r17c/ --android_abi arm64-v8a --android_api 25 "$@"

# --split_UT
