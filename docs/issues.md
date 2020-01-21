
## Trouble Shooting

### Unable to initialize EGL

It is usually caused by incorrect installation of nvidia-driver, insufficient OpenGL version, or no EGL is installed on the machine.

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

If the original installation doesn't work, try the following:

1. Is nvidia driver properly installed? You can check by running nvidia-smi
2. Are openGL libraries visible? You can do so by
`export LD_LIBRARY_PATH=/usr/lib/nvidia-<vvv>:$LD_LIBRARY_PATH`
3. There are two ways of setting up openGL library, if the current installation doesn't work, you can try to install with USE_GLAD set to FALSE in [here](https://github.com/StanfordVL/GibsonEnvV2/blob/master/gibson2/core/render/CMakeLists.txt)

Also, the EGL setup part is borrowed from Erwin Coumans [egl_example](https://github.com/erwincoumans/egl_example). It would be informative to see if that repository can run on your machine.
