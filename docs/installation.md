# Installation
There are two steps to install iGibson, the Interactive Gibson Environment, on your computer.

First, you need to install the simulation environment. Then, you need to download the assets: models of the robotic agents, the interactive objects and 3D reconstructed real-world large environments for your agents to train.

### System Requirements

The minimum system requirements are the following:

- Ubuntu 16.04
- Nvidia GPU with VRAM > 6.0GB
- Nvidia driver >= 384
- CUDA >= 9.0, CuDNN >= v7

Other system configurations may work, but we haven't tested them extensively and we probably won't be able to provide as much support as we want.

## Installing the Environment

We provide 3 methods to install the simulator.

### 1. pip

iGibson's simulator can be installed as a python package using pip:

```bash
pip install gibson2
# run the demo
python -m gibson2.envs.demo
python -m gibson2.envs.demo_interactive
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

Alternatively, iGibson can be compiled from source: [iGibson GitHub Repo](https://github.com/StanfordVL/iGibson)

```bash
git clone https://github.com/StanfordVL/iGibson --recursive
cd iGibson

conda create -n py3-igibson python=3.6 anaconda
source activate py3-igibson
pip install -e .
```
We recommend the third method if you plan to modify iGibson in your project. If you plan to use it as it is to train navigation and manipulation agents, the pip installation or docker image should meet your requirements.


## Downloading the Assets

First, create a folder to contain all the iGibson's assets (robotic agents, objects, 3D environments, etc.) and set the path in `your_installation_path/gibson2/global_config.yaml` (default and recommended: `your_installation_path/gibson2/assets`).

Second, you can download our robot models and objects from [here](https://storage.googleapis.com/gibson_scenes/assets_igibson.tar.gz) and unpack it in the assets folder.

Third, you need to download some large 3D reconstructed real-world environments (houses, offices) from [our dataset](dataset.md) for your agents to be trained in. Create a new folder for those environments and set the path in `your_installation_path/gibson2/global_config.yaml` (default and recommended: `your_installation_path/gibson2/dataset`). You can get access and download the full Gibson and iGibson (interactive furniture) datasets by filling up the following [license agreement](https://forms.gle/YTbzXjNtmmsra9KY6). Alternatively, you can download a single [high quality small environment](https://storage.googleapis.com/gibson_scenes/Rs.tar.gz), R's, together with a [fully interactive version](https://storage.googleapis.com/gibson_scenes/Rs_interactive.tar.gz).

The robot and object models, together with the R's interactive and non-interactive versions can be downloaded and extracted in the assets folder indicated in `your_installation_path/gibson2/global_config.yaml` with two commands:

```bash
python -m gibson2.utils.assets_utils --download_assets
python -m gibson2.utils.assets_utils --download_demo_data
```

The full Gibson and iGibson dataset can be downloaded using the following command, this script automatically download, decompress, and put the dataset to correct place. You will get `URL` after filling in the agreement form.

```bash
python -m gibson2.utils.assets_utils --download_dataset URL
```


### Uninstalling
Uninstalling iGibson is easy: `pip uninstall gibson2`

