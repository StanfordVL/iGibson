#  iGibson: the Interactive Gibson Environment

<img src="./docs/images/igibsonlogo.png" width="500"> <img src="./docs/images/igibson.gif" width="250"> 

### Large Scale Interactive Simulation Environments for Robot Learning

iGibson, the Interactive Gibson Environment, is a simulation environment providing fast visual rendering and physics simulation (based on Bullet). It is packed with a dataset with hundreds of large 3D environments reconstructed from real homes and offices, and interactive objects that can be pushed and actuated. iGibson allows researchers to train and evaluate robotic agents that use RGB images and/or other visual sensors to solve indoor (interactive) navigation and manipulation tasks such as opening doors, picking and placing objects, or searching in cabinets.

### Latest Updates
[12/4/2020] We created a [Slack workspace](https://join.slack.com/t/igibsonuser/shared_invite/zt-jz8x6wgh-2usPj6nMz7mawWyr1tmNfQ) to support iGibson users.

[12/1/2020] Major update to iGibson to reach iGibson v1.0, for details please refer to our [technical report](TBA). 

- Release of iGibson dataset, which consists of 15 fully interactive scenes and 500+ object models.
- New features of the Simulator: Physically based rendering; 1-beam and 16-beam lidar simulation; Domain
 randomization support.
- Code refactoring and cleanup. 

[05/14/2020] Added dynamic light support :flashlight:

[04/28/2020] Added support for Mac OSX :computer:

### Citation
If you use iGibson or its assets and models, consider citing the following publication:

```
@article{shenigibson,
  title={iGibson, a Simulation Environment for Interactive Tasks in Large Realistic Scenes},
  author={Shen, Bokui and Xia, Fei and Li, Chengshu and Mart{\i}n-Mart{\i}n, Roberto and Fan, Linxi and Wang, Guanzhi and Buch, Shyamal and D’Arpino, Claudia and Srivastava, Sanjana and Tchapmi, Lyne P and  Vainio, Kent and Fei-Fei, Li and Savarese, Silvio},
  journal={arXiv preprint},
  year={2020}
}
```


### Release
This is the repository for iGibson (pip package `gibson2`) 1.0 release. Bug reports, suggestions for improvement, as
 well as community
 developments are encouraged and appreciated. The support for our previous version of the environment, [Gibson
  Environment
 ](http://github.com/StanfordVL/GibsonEnv/), will be moved to this repository.

### Documentation
The documentation for this repository can be found here: [iGibson Environment Documentation](http://svl.stanford.edu/igibson/docs/). It includes installation guide (including data download), quickstart guide, code examples, and APIs.

If you want to know more about iGibson, you can also check out [our webpage](http://svl.stanford.edu/igibson),  [our
 updated technical report](TBA) and [our RAL+ICRA20 paper](https://arxiv.org/abs/1910.14442) and.

### Dowloading Dataset of 3D Environments
There are several datasets of 3D reconstructed large real-world environments (homes and offices) that you can download and use with iGibson. All of them will be accessible once you fill in this <a href="https://forms.gle/36TW9uVpjrE1Mkf9A" target="_blank">[form]</a>.

Additionally, with iGibson v1.0 release, you will have access to 15 fully interactive scenes (100+ rooms) that can be
 used in simulation. As a highlight, here
 are the features we support. We also include 500+ object models.  

- Scenes are the
 result of converting 3D reconstructions of real homes into fully interactive simulatable environments.
- Each scene corresponds to one floor of a real-world home.
The scenes are annotated with bounding box location and size of different objects, mostly furniture, e.g. cabinets, doors, stoves, tables, chairs, beds, showers, toilets, sinks...
- Scenes include layout information (occupancy, semantics)
- Each scene's lighting effect is designed manually, and the texture of the building elements (walls, floors, ceilings
) is baked offline with high-performant ray-tracing
- Scenes are defined in iGSDF (iGibson Scene Definition Format), an extension of URDF, and shapes are OBJ files with
 associated materials

For instructions to install iGibson and download dataset, you can visit [installation guide](http://svl.stanford.edu/igibson/docs/installation.html).

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

* SRAnipal (needed for eye tracking):
https://hub.vive.com/en-US/download

Download the VIVE_SRanipalInstaller msi file and install SRAnipal.

Gibson
======

* Get codebase and assets:

```
$ git clone https://github.com/fxia22/iGibson.git --init --recursive
$ cd iGibson
$ git checkout vr_new
$ git submodule update --recursive
```

Follow the instructions on the iGibson website to obtain the iGibson assets and dataset (http://svl.stanford.edu/igibson/docs/).

* Create anaconda env:

```
$ conda create -n gibsonvr python=3.6
```
Activate conda env:
```
$ conda activate gibsonvr
```

* Install Gibson in anaconda env:
```
$ cd iGibson
```
- If you followed the instructions, iGibson is at the vr_new branch
```
$ pip install -e .
```

Important - VR functionality and where to find it:

You can find all the VR demos in iGibson/examples/demo/vr_demos, which has the following structure:

-vr_playground.py

--robot_embodiment (folder)

---vr_demo_robot_control.py

--muvr (folder)

---igvr_client.py

---igvr_server.py

---muvr_demo.py

--data_save_replay (folder)

---vr_states_sr.py

---vr_actions_sr.py

---vr_logs (folder containing saved data)

Additional information:
1) Most VR functions can be found in the gibson2/simulator.py
2) The VrAgent and its associated VR objects can be found in gibson2/objects/vr_objects.py
3) VR utility functions are found in gibson2/utils/vr_utils.py
4) The VR renderer can be found in gibson2/render/mesh_renderer.py
5) The underlying VR C++ code can be found in vr_mesh_render.h and .cpp in gibson2/render/cpp

To get started with the iGibson VR experience run:

$ python vr_playground.py

By default the LOAD_PARTIAL boolean is set to false to speed up loading (loads first 10 objects into the scene as well as some objects to interact with). Please edit this variable to True if you wish to load the entire Rs_int scene.

To use the VR assets, please access the Google drive folder at this link:
https://drive.google.com/drive/folders/1zm3ZpPc7yHwyALEGfsb0_NybFMvV81Um?usp=sharing

You will need to download both vr_body and vr_hand and place them into assets/models. The pack_lunch folder containing the groceries assets used in the ATUS demos can also be found here. Please also put this into your assets/models folder.

Have fun in VR!

Helpful tips:
1) Press ESCAPE to force the fullscreen rendering window to close during program execution (although fullscreen is disabled by default)
2) Before using SRAnipal eye tracking, you may want to re-calibrate the eye tracker. Please go to the Vive system settings to perform this calibration.
