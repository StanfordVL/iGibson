#  iGibson: A Simulation Environment to train Robots in Large Realistic Interactive Scenes

<img src="./docs/images/igibsonlogo.png" width="500"> <img src="./docs/images/igibson.gif" width="250"> 

iGibson is a simulation environment providing fast visual rendering and physics simulation based on Bullet. iGibson is equipped with fifteen fully interactive high quality scenes, hundreds of large 3D scenes reconstructed from real homes and offices, and compatibility with datasets like CubiCasa5K and 3D-Front, providing 8000+ additional interactive scenes. Some of the features of iGibson include domain randomization, integration with motion planners and easy-to-use tools to collect human demonstrations. With these scenes and features, iGibson allows researchers to train and evaluate robotic agents that use visual signals to solve navigation and manipulation tasks such as opening doors, picking up and placing objects, or searching in cabinets.

### Latest Updates

[12/1/2020] Major update to iGibson to reach iGibson v1.0, for details please refer to our [arxiv preprint](https://arxiv.org/abs/2012.02924). 

- Release of iGibson dataset that includes 15 fully interactive scenes and 500+ object models annotated with materials and physical attributes on top of [existing 3D articulated models](https://cs.stanford.edu/~kaichun/partnet/).
- Compatibility to import [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) and [3D-Front](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) scene descriptions leading to more than 8000 extra interactive scenes!
- New features in iGibson: Physically based rendering, 1-beam and 16-beam LiDAR, domain randomization, motion planning integration, tools to collect human demos and more!
- Code refactoring, better class structure and cleanup. 

[05/14/2020] Added dynamic light support :flashlight:

[04/28/2020] Added support for Mac OSX :computer:

### Citation
If you use iGibson or its assets and models, consider citing the following publication:

```
@article{shenigibson,
  title={iGibson, a Simulation Environment for Interactive Tasks in Large Realistic Scenes},
  author={Shen*, Bokui and Xia*, Fei and Li*, Chengshu and Mart{\'i}n-Mart{\'i}n*, Roberto and Fan, Linxi and Wang, Guanzhi and Buch, Shyamal and D’Arpino, Claudia and Srivastava, Sanjana and Tchapmi, Lyne P and  Vainio, Kent and Fei-Fei, Li and Savarese, Silvio},
  journal={arXiv preprint arXiv:2012.02924},
  year={2020}
}
```

### Documentation
The documentation for iGibson can be found here: [iGibson Documentation](http://svl.stanford.edu/igibson/docs/). It includes installation guide (including data download instructions), quickstart guide, code examples, and APIs.

If you want to know more about iGibson, you can also check out [our webpage](http://svl.stanford.edu/igibson),  [our
 updated arxiv preprint](https://arxiv.org/abs/2012.02924) and [our previous RAL+ICRA20 paper](https://arxiv.org/abs/1910.14442).

### Dowloading the Dataset of 3D Scenes

With iGibson v1.0 release, you will have access to 15 fully interactive scenes (100+ rooms) that can be
 used in simulation. As a highlight, here
 are the features we support. We also include 500+ object models.  

- Scenes are the
 result of converting 3D reconstructions of real homes into fully interactive simulatable models.
- Each scene corresponds to one floor of a real-world home.
The scenes are annotated with bounding box location and size of different objects, mostly furniture, e.g. cabinets, doors, stoves, tables, chairs, beds, showers, toilets, sinks...
- Scenes include layout information (occupancy, semantics)
- Each scene's lighting effect is designed manually, and the texture of the building elements (walls, floors, ceilings
) is baked offline with high-performant ray-tracing
- Scenes are defined in iGSDF (iGibson Scene Definition Format), an extension of URDF, and shapes are OBJ files with
 associated materials

For instructions to install iGibson and download dataset, you can visit [installation guide](http://svl.stanford.edu/igibson/docs/installation.html).

There are other datasets we link to iGibson. We include support to use CubiCasa5K and 3DFront scenes, adding up more than 8000 extra interactive scenes to use in iGibson! Check our documentation on how to use those.

We also maintain compatibility with datasets of 3D reconstructed large real-world scenes (homes and offices) that you can download and use with iGibson, for example from our previous simulator, Gibson. All of them will be accessible once you fill in this <a href="https://forms.gle/36TW9uVpjrE1Mkf9A" target="_blank">[form]</a>.

### Contributing
This is the github repository for iGibson (pip package `gibson2`) 1.0 release. Bug reports, suggestions for improvement, as well as community developments are encouraged and appreciated. Please, consider creating an issue or sending us an email. 

The support for our previous version of the environment, Gibson, can be found in the [following repository](http://github.com/StanfordVL/GibsonEnv/).

# VR Information

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

You can find all the VR demos in iGibson/gibson2/examples/demo/vr_demos, which has the following structure:

- vr demo files for normal VR, multi-user VR and robot embodiment VR

- data_save_replay folder containing save/replay demos

- atus folder containing lunch packing demo

Additional information:
1) Most VR functions can be found in the gibson2/simulator.py
2) The VrAgent and its associated VR objects can be found in gibson2/objects/vr_objects.py
3) VR utility functions are found in gibson2/utils/vr_utils.py
4) The VR renderer can be found in gibson2/render/mesh_renderer.py
5) The underlying VR C++ code can be found in vr_mesh_render.h and .cpp in gibson2/render/cpp

To get started with the iGibson VR experience run:

$ python vr_playground.py

By default the LOAD_PARTIAL boolean is set to false to speed up loading (loads first 10 objects into the scene as well as some objects to interact with). Please edit this variable to True if you wish to load the entire Rs_int scene.

Note: we recommend looking at gibson2/render/mesh_renderer/mesh_renderer_vr.py to see the VrSettings class. A single VrSettings object is created and passed in to the Simulator constructor at the start of every demo, and you can modify it to change your experience. Here is a list of the things you can set, along with their default values:

  1) use_vr (default True) - whether to render to the HMD and use VR system or just render to screen (used for debugging)
  2) eye_tracking (default True) - whether to use eye tracking. Turn this off if your headset does not support eye tracking (only the HTC Vive Pro Eye does)
  3) touchpad_movement (default True) - whether to enable use of touchpad to move - this will help you get around large scenes if your play area is small
  4) movement_controller (default 'right') - device to controler movement - can be right or left (representing the corresponding controllers). Default is for right-handed people - please change to left if that is more comfortable.
  4) relative_movement_device (default 'hmd') - which device to use to control touchpad movement direction (can be any VR device). You should not need to change this.
  5) movement_speed (default 0.01) - how fast you move when using the touchpad. This number has been calibrated carefully, however feel free to change it if you want to speed up/slow down.

To use the VR assets, please access the Google drive folder at this link:
https://drive.google.com/drive/folders/1zm3ZpPc7yHwyALEGfsb0_NybFMvV81Um?usp=sharing
Please place all of these folders into your assets/models folder, with their original names.

Have fun in VR!

Helpful tips:
1) Press ESCAPE to force the fullscreen rendering window to close during program execution (although fullscreen is disabled by default)
2) Before using SRAnipal eye tracking, you may want to re-calibrate the eye tracker. Please go to the Vive system settings to perform this calibration.
