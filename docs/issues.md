
## Troubleshooting

### Installation Issues

#### ```ModuleNotFoundError: No module named 'cmake'```

This problem appears often on MacOS. You should have CMake installed in your computer **system-wide**. On Mac, you can use brew to install it: ``brew install cmake`` (search how to install the package manager Homebrew if it is the first time you use it). If you use cmake installed with pip (`pip install cmake`) or in conda (`conda install cmake`), this installation issue may appear.

#### `.../NSBundle.h:91:143: error: function does not return NSString`

This problem appears in MacOS. The entire error message should be something like: 
```
/Library/Developer/CommandLineTools/SDKs/MacOSX12.1.sdk/System/Library/Frameworks/Foundation.framework/Headers/NSBundle.h:91:143: error: function does not return NSString
```

This error is caused by a change in the Command-line tools (CLT) of MacOS. CLT can be installed as part of Xcode or as stand-alone suite. If you have Xcode, consider downgrading or upgrading it to a different version. If you do not have Xcode, we have seen success on Big Sur by installing an older version of the CLT:
1. Removing your current version of Command-line tools following the instructions [here](https://mac.install.guide/commandlinetools/6.html)
2. Grabbing an older version. On Big Sur (11.6), you can use the version 12.0 (see that the error message indicated 12.1). Go to [this page](https://developer.apple.com/download/all/?q=command%20line%20tools), login with your itunes account, download and install the older version. Recompile.

However, compilation is again successful with newer versions of Xcode/CLT as well (tested with Xcode.

#### `tried: 'XXXXX.so' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e'))`

We have observed this error when compiled on MacOS with an old version of CMake. Consider installing systemwide a newer version of CMake, e.g., using Homebrew, to a version > 2.17.

#### `ERROR: Could not build wheels for h5py, which is required to install pyproject.toml-based projects`

This error has been observed on arm64 MacOS. 

Solution: `conda install h5py`

### Execution Issues

#### `RuntimeError: Freetype library not found`

This error has been observed on arm64 MacOS. The pip package `freetype-py` may install binaries for the wrong architecture. 

Solution: `conda install freetype`

#### Unable to initialize EGL

1. Is nvidia driver properly installed? Nvidia drivers prior to 460.80 and 465.27 have a bug with cgroups that prevent utilize all but GPU 0 on a node. You can check your driver version by running nvidia-smi and update if necessary.
2. Is libegl1 installed? You can determine this by `apt list --installed | grep libegl1`
3. Are openGL libraries visible? You can do so by
`export LD_LIBRARY_PATH=/usr/lib/nvidia-<vvv>:$LD_LIBRARY_PATH`
4. There are two ways of setting up openGL library, if the current installation doesn't work, you can try to install with USE_GLAD set to FALSE in [here](https://github.com/StanfordVL/iGibson/blob/master/igibson/render/CMakeLists.txt)
5. If you want to render in headless mode, make sure `$DISPLAY` environment variable is unset, otherwise you might have error `Failed to EGL with glad`, because EGL is sensitive to `$DISPLAY` environment variable.

It is a good idea to run `ldconfig -p | grep EGL` and you should be able to see `libEGL_nvidia` installed.

```
	libEGL_nvidia.so.0 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0
	libEGL_nvidia.so.0 (libc6) => /usr/lib/i386-linux-gnu/libEGL_nvidia.so.0
	libEGL_mesa.so.0 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0
	libEGL.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libEGL.so.1
	libEGL.so.1 (libc6) => /usr/lib/i386-linux-gnu/libEGL.so.1
	libEGL.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libEGL.so
	libEGL.so (libc6) => /usr/lib/i386-linux-gnu/libEGL.so
```

Make sure EGL is not linked to mesa, because in order for gibson to work, linking to Nvidia's EGL is required. For example, this will cause problems: `/usr/lib/x86_64-linux-gnu/libEGL.so -> mesa-egl/libEGL.so`

The EGL setup part is borrowed from Erwin Coumans [egl_example](https://github.com/erwincoumans/egl_example). It would be informative to see if that repository can run on your machine.

### Other issues
For other issues, please submit an issue in our [github repository](https://github.com/StanfordVL/iGibson/issues). 
