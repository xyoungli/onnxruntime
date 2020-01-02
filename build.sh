#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set -e

# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_TYPE="MinSizeRel" # "Debug", "MinSizeRel", "Release", "RelWithDebInfo"
BUILD_DIR=$DIR/../build/Linux/

python3 $DIR/tools/ci_build/build.py \
  --skip_submodule_sync --skip_tests \
  --config $BUILD_TYPE --parallel \
  --use_openmp --build_dir $BUILD_DIR "$@"

