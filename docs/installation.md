# Installation
There are two steps to install iGibson, the Interactive Gibson Environment, on your computer.

First, you need to install the simulation environment. Then, you need to download the assets: models of the robotic agents, the interactive objects and 3D reconstructed real-world large environments for your agents to train.

### System Requirements

The minimum system requirements are the following:

- Linux
    - Ubuntu 16.04
    - Nvidia GPU with VRAM > 6.0GB
    - Nvidia driver >= 384
    - CUDA >= 9.0, CuDNN >= v7
    - CMake >= 2.8.12 (can install with `pip install cmake`)
- Mac OS X
    - Tested on 10.15
    - PBR features not supported
    - CMake >= 2.8.12 (can install with `pip install cmake`)
- Windows
    - Coming soon

Other system configurations may work, but we haven't tested them extensively and we probably won't be able to provide as much support as we want.

## Installing the Environment

We provide 3 methods to install the simulator.

### 1. pip

iGibson's simulator can be installed as a python package using pip:

```bash
pip install gibson2  # This step takes about 4 minutes
# run the demo
python -m gibson2.examples.demo.demo_static
```

Note: we support using a custom pybullet version to speed up the physics in iGibson, if you want to have the speed up, you would need to do the following steps after installation:

```bash
pip uninstall pybullet
pip install https://github.com/StanfordVL/bullet3/archive/master.zip
```

### 2. Docker image

Docker provides an easy way to reproduce the development environment across platforms without manually installing the software dependencies. We have prepared docker images that contain everything you need to get started with iGibson.  

First, install Docker from the [official website](https://www.docker.com/). Please make sure that the docker version is at least v19.0 to enable native GPU support.

Next, download our pre-built images with the script in the `iGibson` repo:

```
cd iGibson
./docker/pull-images.sh
```

Two images will be downloaded:
* `igibson/igibson:latest`: smaller image, but does not support GUI. 
* `igibson/igibson-gui:latest`: supports GUI and remote desktop access via VNC.

We also provide scripts to build the images from scratch:
```
# image without GUI:
cd iGibson/docker/base
./build.sh

# image with GUI and VNC:
cd iGibson/docker/headless-gui
./build.sh
```


### 3. Compile from source

Alternatively, iGibson can be compiled from source: [iGibson GitHub Repo](https://github.com/StanfordVL/iGibson). First, you need to install anaconda following the guide on [their website](https://www.anaconda.com/). 

```bash
git clone https://github.com/StanfordVL/iGibson --recursive
cd iGibson

conda create -n py3-igibson python=3.6 anaconda # we support python 3.5, 3.6, 3.7, 3.8
source activate py3-igibson
pip install -e . # This step takes about 4 minutes
```

Note: we support using a custom pybullet version to speed up the physics in iGibson, if you want to have the speed up, you would need to do the following steps after installation:

```bash
pip uninstall pybullet
pip install https://github.com/StanfordVL/bullet3/archive/master.zip
```

We recommend the third method if you plan to modify iGibson in your project. If you plan to use it as it is to train navigation and manipulation agents, the pip installation or docker image should meet your requirements.


## Downloading the Assets

First, configure where iGibson's assets (robotic agents, objects, 3D environments, etc.) is going to be stored. It is configured in `your_installation_path/gibson2/global_config.yaml`

To make things easier, the default place to store the data is:
```bash
assets_path: your_installation_path/gibson2/data/assets 
g_dataset_path: your_installation_path/gibson2/data/g_dataset
ig_dataset_path: your_installation_path/gibson2/data/ig_dataset
threedfront_dataset_path: your_installation_path/gibson2/data/threedfront_dataset 
cubicasa_dataset_path: your_installation_path/gibson2/data/assetscubicasa_dataset 
```

If you are happy with the default path, you don't have to do anything, otherwise you can run this script:
```bash
python -m gibson2.utils.assets_utils --change_data_path
```

Second, you can download our robot models and objects from [here](https://storage.googleapis.com/gibson_scenes/assets_igibson.tar.gz) and unpack it in the assets folder, or simply run this download script:

```bash
python -m gibson2.utils.assets_utils --download_assets
```


Third, you need to download some large 3D reconstructed real-world environments (e.g. houses and offices) from [our dataset](dataset.md) for your agents to be trained in. Create a new folder for those environments and set the path in `your_installation_path/gibson2/global_config.yaml` (default and recommended: `your_installation_path/gibson2/data/g_dataset` and `your_installation_path/gibson2/data/ig_dataset`). You can get access and download the Gibson and iGibson datasets by filling up the following [license agreement](https://forms.gle/36TW9uVpjrE1Mkf9A). In addition, you can download a single [high quality small environment R's](https://storage.googleapis.com/gibson_scenes/Rs.tar.gz) for demo purposes.

To download the demo data, run:

```bash
python -m gibson2.utils.assets_utils --download_demo_data
```

The full Gibson and iGibson dataset can be downloaded using the following command, this script automatically downloads, decompresses, and puts the dataset to correct place. You will get `URL` after filling in the agreement form.

Download iGibson dataset
```bash
python -m gibson2.utils.assets_utils --download_ig_dataset
```

Download Gibson dataset ([agreement signing](https://forms.gle/36TW9uVpjrE1Mkf9A) required to get `URL`)
```bash
python -m gibson2.utils.assets_utils --download_dataset URL
```

## Testing 

To test gibson2 is properly installed, you can run 
```bash
python
>> import gibson2
```

For a full suite of tests and benchmarks, you can refer to [tests](tests.md) for more details. 

## Uninstalling
Uninstalling iGibson is easy: `pip uninstall gibson2`
