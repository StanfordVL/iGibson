#!/bin/bash
# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Change working directory to script folder
SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "${SCRIPT_DIR}"


PROFILE="Release"
VERBOSE_MAKE=""

declare -a BUILD_FLAGS
declare -a CONFIG_FLAGS

ANDROID_NDK="~/android-ndk-r15c/"
ANDROID_NATIVE_API_LEVEL="21"
ANDROID_ABI="armeabi-v7a with NEON"

MSVC_GENERATOR="Visual Studio 16 2019"

function show_help()
{
  cat <<EOF
*** Resonance Audio compilation script ***

Please select a build target:
  -p= | --profile=[Debug|Release], default: Release
  --msvc_dynamic_runtime     # Enables dynamic runtime environment linking in MSVC builds.
EOF
exit
}

BUILD_TARGET="IGIBSON_PLUGIN"

for i in "$@"
do
  case $i in
    -p=*|--profile=*)
      PROFILE="${i#*=}"
      shift # past argument=value
      ;;

    -t=*|--target=*)
      BUILD_TARGET="${i#*=}"
      shift # past argument=value
      ;;

    --msvc_dynamic_runtime)
      CONFIG_FLAGS+=(-DSTATIC_MSVC_RUNTIME:BOOL=OFF)
      shift # past argument with no value
      ;;

    *)
      # unknown option
      echo "Unknown option: ${i}"
      show_help
      ;;
  esac
done

[[ -z "$BUILD_TARGET" ]] && show_help

# Number of CPU cores/parallel compilation instances (for Darwin/Linux builds)
NUM_CORES=8

# Create build environment.
rm -fr build && mkdir build && cd build

case "$(uname -s)" in
  Darwin)
    BUILD_FLAGS+=(-j "${NUM_CORES}")
    cmake -DBUILD_"${BUILD_TARGET}":BOOL=ON\
      "${CONFIG_FLAGS[@]}" "$@" ..
    ;;

  Linux)
    BUILD_FLAGS+=(-j "${NUM_CORES}")
    cmake -DBUILD_"${BUILD_TARGET}":BOOL=ON\
      "${CONFIG_FLAGS[@]}" "$@" ..
    ;;

  CYGWIN*|MINGW*|MSYS*)
    cmake -G"${MSVC_GENERATOR}"\
      -DBUILD_"${BUILD_TARGET}":BOOL=ON\
      "${CONFIG_FLAGS[@]}" "$@" ..
    ;;

  *)
    ;;
esac

INSTALL_TARGET="install"
if echo "${BUILD_TARGET}" | grep -q "TESTS"; then
  INSTALL_TARGET=""
fi

cmake --build . --config "${PROFILE}" --target "${INSTALL_TARGET}" -- "${BUILD_FLAGS[@]}"
