
# VR Information

Instructions for installing Gibson and VR integration on Windows 10.
Assuming a fresh install of Windows.

These instructions partially overlap with installing Gibson
http://svl.stanford.edu/igibson/docs/installation.html#installation-method 
but are tailored to run the VR components in Windows.


VR Station
==========

Install Steam and Steam VR, connect VR headset and base stations, set up VR room
Run steam performance test.

https://www.vive.com/eu/support/vive/category_howto/setting-up-for-the-first-time.html


Dependencies and environment:
=============================

* Skip this step if you have a working terminal setup in Windows

Follow these instructions to download and install Cygwin: https://www.cygwin.com/ and then https://www.howtogeek.com/howto/41382/how-to-use-linux-commands-in-windows-with-cygwin/ - Cygwin provides a similar experience to the Linux terminal on Windows, and makes working with iGibson much easier.

**Note:** In the article they add C:\Cygwin\bin to the PATH, but the name of your folder may differ. Check in your C: drive in file explorer to see whether you have a folder named Cygwin, or cygwin64 (or perhaps something different). Use this path instead.

**Pro tip:** To run the Windows command line, press Win+R, then type cmd. We recommend pinning this to your taskbar for easy access.

* git 
https://git-scm.com/download/win

* Python
https://www.python.org/downloads/release/python-377/

* Anaconda 
https://www.anaconda.com/distribution/#download-section

Make sure anaconda is added to the PATH by ticking the appropriate box during installation.

* Build Tools for Visual Studio:
Microsoft Visual C++ 14.0 is required. Get it with "Build Tools for Visual Studio": 
https://visualstudio.microsoft.com/downloads/
This is needed for bullet

* cmake:
https://cmake.org/download/
Needed for building MeshRendererContext and Pybind.

* SRAnipal (needed for eye tracking):
https://hub.vive.com/en-US/download

Download the VIVE_SRanipalInstaller msi file and install SRAnipal.

Companion Window:
=============================

iGibson VR utilizes Steam VR's built-in companion window to visualize what the user sees in their headset. To open this window, launch Steam VR. Then click on the hamburger menu in the top-left corner of the VR status menu (the dark blue window with icons for the VR devices).
Within that menu, locate the "Display VR View" button and click it. From this view, you can change which eye you are looking at (or look at both), and can make the window fullscreen.
Note that this window will be black until an application is running, and the headset is detected by the lighthouse sensors.

We also support a custom-build companion window that can run in iGibson - this can be enabled in the vr_config file, described below (although it is off by default).

Gibson
======

* Get codebase and assets:

```
$ git clone https://github.com/fxia22/iGibson.git --recursive (depending on your computer you may need to add the --init flag as well)
$ cd iGibson
$ git fetch origin
$ git checkout -b igvr origin/igvr
$ git submodule update --recursive
```

Follow the instructions on the iGibson website to obtain the iGibson assets and dataset (http://svl.stanford.edu/igibson/docs/).

* Create anaconda env:

```
$ conda create -n igvr python=3.6
```
Activate conda env:
```
$ conda activate igvr
```
* Install Gibson in anaconda env:
```
$ cd iGibson
```
- Make sure you are on the igvr branch.
```
$ pip install -e .
```

Now, with your conda environment activate, run the following two commands:
1) pip uninstall pybullet && pip install --no-cache-dir https://github.com/StanfordVL/bullet3/archive/master.zip (this replaces Stanford pybullet with SVL's pybullet, which contains optimizations for VR)
2) pip install podsixnet (networking library used in multi-user VR)

Important - VR functionality and where to find it:

You can find all the VR demos in iGibson/igibson/examples/demo/vr_demos, which has the following structure:

- vr demo files for normal VR

- atus folder containing lunch packing demo

- data_save_replay folder containing save/replay demos

- muvr folder containing multi-user VR demos

- robot_embodiment folder containing robot embodiment VR demos

Benchmarks:

We have two benchmarks - a hand and speed benchmark, both of which can be found in the top leve of the vr_demos folder. In these demos, you can time yourself performing various challenges,
such as putting objects away into specific containers/cabinets. Please see the comments in these demo files for more information.

VR settings and button mapping:

You can find the global vr settings in the vr_config.yaml in the igibson folder. Please change all VR settings from here. Within the settings, you will also find the button mapping from actions to buttons.
Please run vr_demos/vr_button_mapping.py to figure out which physical controller buttons correspond to which indices in OpenVR. It is only necessary to do this if you are remapping an existing controller,
or adding in a mapping for a new controller.

Additional information:
1) Most VR functions can be found in the igibson/simulator.py
2) The VrAgent and its associated VR objects can be found in igibson/objects/vr_objects.py
3) VR utility functions are found in igibson/utils/vr_utils.py
4) The VR renderer can be found in igibson/render/mesh_renderer.py
5) The underlying VR C++ code can be found in vr_mesh_render.h and .cpp in igibson/render/cpp

To get started with the iGibson VR experience run:

$ python vr_playground.py

By default the LOAD_PARTIAL boolean is set to false to speed up loading (loads first 10 objects into the scene as well as some objects to interact with). Please edit this variable to True if you wish to load the entire Rs_int scene.

Note: we recommend looking at igibson/render/mesh_renderer/mesh_renderer_vr.py to see the VrSettings class. A single VrSettings object is created and passed in to the Simulator constructor at the start of every demo, and you can modify it to change your experience. Here is a list of the things you can set, along with their default values:

  1) use_vr (default True) - whether to render to the HMD and use VR system or just render to screen (used for debugging)
  2) eye_tracking (default True) - whether to use eye tracking. Turn this off if your headset does not support eye tracking (only the HTC Vive Pro Eye does)
  3) touchpad_movement (default True) - whether to enable use of touchpad to move - this will help you get around large scenes if your play area is small
  4) movement_controller (default 'right') - device to controler movement - can be right or left (representing the corresponding controllers). Default is for right-handed people - please change to left if that is more comfortable.
  4) relative_movement_device (default 'hmd') - which device to use to control touchpad movement direction (can be any VR device). You should not need to change this.
  5) movement_speed (default 0.01) - how fast you move when using the touchpad. This number has been calibrated carefully, however feel free to change it if you want to speed up/slow down.

To use the VR assets, please access the Google drive folder at this link:
https://drive.google.com/drive/folders/1zm3ZpPc7yHwyALEGfsb0_NybFMvV81Um?usp=sharing
Please place all of these folders into your assets/models folder, with their original names. Place the fonts folder in the top-level assets directory.

Have fun in VR!

Helpful tips:
1) Press ESCAPE to force the fullscreen rendering window to close during program execution (although fullscreen is disabled by default)
2) Before using SRAnipal eye tracking, you may want to re-calibrate the eye tracker. Please go to the Vive system settings to perform this calibration.
