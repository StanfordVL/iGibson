# [Resonance Audio](https://developers.google.com/resonance-audio) Source Code [![Travis CI](https://travis-ci.org/resonance-audio/resonance-audio.svg?branch=master)](https://travis-ci.org/resonance-audio/resonance-audio)

This is a modified version of the Resonance Audio SDK. This
repository consists of the full source code of the Resonance Audio C++ library,
as well as the platform integration into iGibson.

In this document there are some quick instructions for how to build the SDK from
source code.

For more detailed documentation about using the SDK, visit the [developer
docs](https://developers.google.com/resonance-audio/).


## Build Instructions

Clone the repository:

    git clone https://github.com/resonance-audio/resonance-audio $YOUR_LOCAL_REPO

### Software Requirements

In addition to the system C++ software development platform tools / toolchains,
the following software is required to build and install the Resonance Audio
SDKs:

-   [CMake](https://cmake.org/download/)
-   [Git (with Git-Bash on Windows)](https://git-scm.com/downloads)

_Note: For Windows builds, [Visual Studio
2015](https://www.visualstudio.com/vs/older-downloads/) is recommended._

### Third Party Dependencies

All third party dependencies must be installed into the `third_party` subfolder
in the repository. To simplify the installation, bash scripts are located within
the `third_party` that automatically clone, build and install the required third
party source code.

_Note: On Windows, these scripts can be executed in the Git-Bash console (which
gets installed as part of Git for Windows)._

#### Core Dependencies ([pffft](https://bitbucket.org/jpommier/pffft), [eigen](https://bitbucket.org/eigen/eigen), [googletest](https://github.com/google/googletest), [SADIE Binaural Measurements](https://www.york.ac.uk/sadie-project/database_old.html))

To clone the dependencies into the repository, run:

    ./$YOUR_LOCAL_REPO/third_party/clone_core_deps.sh

_Note: These dependencies do *not* need to be built, since their source code is
directly pulled in from the build scripts._

#### [Unity](https://unity3d.com/) Platform Dependencies ([nativeaudioplugins](https://github.com/Unity-Technologies/NativeAudioPlugins), [embree](https://github.com/embree/embree), [ogg](https://github.com/xiph/ogg), [vorbis](https://github.com/xiph/vorbis))

The Unity plugin integrates additional tools to estimate reverberation from game
geometry and to capture Ambisonic soundfields from a game scene. These features
require the Embree, libOgg and libVorbis libraries to be *prebuilt*.

To clone and build the additional Unity dependencies, run:
    ./$YOUR_LOCAL_REPO/third_party/clone_build_install_unity_deps.sh


### Build Resonance Audio SDKs

This repository provides the `build.sh` script in the root folder that
configures the build targets, triggers the compilation and installs the
artifacts for the specified platform into the target installation folder.

The script provides the following flags:

-   t=|--target=
    -   `RESONANCE_AUDIO_API`: Builds the Resonance Audio API
    -   `RESONANCE_AUDIO_TESTS`: Runs the Resonance Audio unit tests
    -   `GEOMETRICAL_ACOUSTICS_TESTS`: Runs the geometrical acoustics specific
        unit tests.
    -   `IGIBSON_PLUGIN`: Builds the Resonance Audio plugin for Unity
-   p=|--profile=
    -   `Debug`: Debug build
    -   `RelWithDebInfo`: Release build with debug information
    -   `Release`: Release build
-   --msvc_dynamic_runtime
    -   Enables dynamic linking against the run-time library on Windows (`/MD`,
        `/MDd`). By default, all Windows builds are statically linked against
        the run-time library (`/MT`, `/MTd`). Note that the third party
        dependencies must be compiled with the same options to avoid library
        conflicts.
-   --verbose_make
    -   Enables verbose make/build output.

##### E.g.

To build and run the Resonance Audio unit tests:
    ./$YOUR_LOCAL_REPO/build.sh -t=RESONANCE_AUDIO_TESTS

## Citations

If you find Resonance Audio useful and would like to cite it in your publication, please use:

Gorzel, M., Allen, A., Kelly, I., Kammerl, J., Gugnormusler, A., Yeh, H., and Boland, F., *"Efficient Encoding and Decoding of Binaural Sound with Resonance Audio"*, In proc. of the AES International Conference on Immersive and Interactive Audio, March 2019

The full paper is available (open access) at: http://www.aes.org/e-lib/browse.cfm?elib=20446 ([BibTeX](http://www.aes.org/e-lib/browse.cfm?elib=20446&fmt=bibtex))
