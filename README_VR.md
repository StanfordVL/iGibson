# Gibson in VR

Instructions for installing Gibson and VR integration on Windows 10.
Assuming a fresh install of Windows.

These instructions partially overlap with installing Gibson
http://svl.stanford.edu/gibson2/docs/installation.html#installation-method 
but are tailored to run the VR components in Windows.


VR Station
==========

Install Steam and Steam VR, connect VR headset and base stations, set up VR room
Run steam performance test.

https://www.vive.com/eu/support/vive/category_howto/setting-up-for-the-first-time.html


Dependencies and environment:
=============================

* git 
https://git-scm.com/download/win

* Python
https://www.python.org/downloads/release/python-377/

* Anaconda 
https://www.anaconda.com/distribution/#download-section

Make sure anaconda is added to the PATH as follows:
C:\Users\C\anaconda3
C:\Users\C\anaconda3\Scripts
C:\Users\C\anaconda3\Library\bin

Lack of the latter produced the following error:
HTTP 000 CONNECTION FAILED for url <https://repo.anaconda.com/pkgs/main/win-64/current_repodata.json> Elapsed

* Build Tools for Visual Studio:
Microsoft Visual C++ 14.0 is required. Get it with "Build Tools for Visual Studio": 
https://visualstudio.microsoft.com/downloads/
This is needed for bullet

* cmake:
https://cmake.org/download/
Needed for building MeshRendererContext and Pybind.


Gibson
======

* Get codebase and assets:

```
$ git clone git@github.com:fxia22/iGibson.git
$ cd iGibson
$ git checkout vr
$ git submodule update --init --recursive
```

After this you should have content at:
core/render/glfw
core/render/pybind11
core/render/openvr

Download Gibson assets and copy to iGibson/gibson2/assets/
Download enviroments (scenes) and copy to iGibson/gibson2/assets/dataset

* Create anaconda env:

```
$ conda create -n gibsonvr python=3.6
```
Activate conda env:
```
$ source activate gibsonvr
```

* Install Gibson in anaconda env:
```
$ cd iGibson
```
- If you followed the instructions, iGibson is at the vr branch
```
$ pip install -e .
```

Should end printing 'Successfully installed gibson2'

You can find VR tests in iGibson/test and VR demos iGibson/examples/demo

* Example: run the stadium interaction demo in the folder iGibson/examples/demo:

$ python vr_interaction_demo_stadium.py

Have fun in VR!