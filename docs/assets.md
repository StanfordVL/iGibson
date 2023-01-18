# Assets

## Introduction

The iGibson assets includes the necessary files to create simulation scenes and other utilities: robot URDF models, models of some interactive objects and articulated objects for testing, copies of the [RBO](https://tu-rbo.github.io/articulated-objects/) and [YCB](http://www.ycbbenchmarks.com/object-models/) datasets of objects to include them in the scenes, a copy of the neural network filler developed in Gibson V1, and some mesh files used to perform installation tests. Due to their large size, these files are not distributed as part of our pip/git packages, but as separate packages. 

The assets file can be downloaded and decompressed to the path set in `your_installation_path/igibson/global_config.yaml` (default to `your_installation_path/igibson/data/assets`) by executing:

```bash 
python -m igibson.utils.assets_utils --download_assets
```

The folder structure will look like below:
```
assets
├── models
│   ├── ant
│   ├── atlas
│   ├── botlab
│   ├── cabinet
│   ├── cabinet2
│   ├── cassie
│   ├── dust
│   ├── fetch
│   ├── fire
│   ├── humanoid
│   ├── husky
│   ├── jr1_urdf
│   ├── jr2_urdf
│   ├── kinova
│   ├── laikago
│   ├── locobot
│   ├── mjcf_primitives
│   ├── person_meshes
│   ├── quadrotor
│   ├── quadruped
│   ├── rbo
│   │    └── ...
│   ├── scene_components
│   │    └── ...
│   ├── stain
│   ├── turtlebot
│   ├── vr_agent
│   └── ycb
│        └── ...
├── networks
│   └── model.pth
└── test
    └── mesh
```

In the following, we include some details of what is included in each subfolder: 'models', 'networks', and 'test'.

## Robot and Object Models (`models`)

URDF and mesh models of robots and objects to load in iGibson scenes.

While we include many robot URDF models, only a subset of those are currently available in iGibson through a corresponding python class (see [robots](./robots.md)). Users could create their own robot classes for the missing URDF models.

We also include [YCB objects](http://www.ycbbenchmarks.com/object-models/), [RBO objects](https://tu-rbo.github.io/articulated-objects/), and a few simple objects commonly used in home environments such as doors (in `scene_components`) and cabinets (in `cabinet` and `cabinet2`). You can refer to [objects](./objects.md) page to see how to use these models in gibson scenes. Don't forget to cite the corresponding YCB and/or RBO publications if you use these assets in your experiments.

## Pretrained network (`networks`)

`networks` folder contains a pretrained network checkpoint `model.pth`, which is the "goggle" network used in Gibson V1. It is used to fix artifacts caused by imperfect reconstruction. In previous version we used image based rendering and there are more visible artifacts, therefore the "goggle" network is important. In this version, mesh rendering creates photorealstic visuals, so we recommend not using it (to gain more framerate). 

## Test meshes (`test`)

In `test` folder we save a mesh file to test the renderer is compiled correctly, it is used by `test/test_render.py`.
