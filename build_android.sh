#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set -e

# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

BUILD_TYPE="MinSizeRel" # "Debug", "MinSizeRel", "Release", "RelWithDebInfo"
ABI="armv8"
ANDROID_TOOLCHAIN="clang"
#ABI="armv7"
#ANDROID_TOOLCHAIN="clang"

BUILD=build/Linux/$ABI/$ANDROID_TOOLCHAIN
BUILD_DIR=$DIR/../$BUILD

#requires python3.6 or higher
#--build_shared_lib
python3 $DIR/tools/ci_build/build.py \
  --skip_submodule_sync --parallel \
  --use_openmp --config $BUILD_TYPE \
  --build_dir $BUILD_DIR \
  --android --android_ndk_path /home/android-ndk-r17c/ \
  --android_abi $ABI --android_toolchain $ANDROID_TOOLCHAIN --android_api 25 "$@"
