
Installation
=================

There are two steps to get iGibson, the Interactive Gibson Environment, on your computer. First, you need to install the simulation environment. Then, you need to download some 3D reconstructed large environments for your agents to train.

#### Installing the Environment

We provide two methods to install iGibson.

First, iGibson can be installed as a python package using pip:

```bash
pip install gibson2
# run the demo
python -m gibson2.envs.demo
python -m gibson2.envs.demo_interactive
```

Alternatively, it can be compiled from source based on this repository:

```bash
git clone https://github.com/StanfordVL/iGibson --recursive
cd iGibson

conda create -n py3-gibson python=3.6 anaconda
source activate py3-gibson
pip install -e .
```
We recommend the second method if you plan to modify iGibson for your project. If you plan to use it as-is to train navigation and manipulation agents, the pip installation should meet your requirements.

#### System Requirements

The minimum system requirements are the following:

- Ubuntu 16.04
- Nvidia GPU with VRAM > 6.0GB
- Nvidia driver >= 384
- CUDA >= 9.0, CuDNN >= v7

Other system configurations may work, but haven't been extensively tested and we won't be able to provide support. We have been able to install on Windows.

#### Download Assets: Models of Robots, Objects and Environments

First, create a folder to contain all the assets (robotic agents, objects, 3D environments, etc.) and set the path in `your_installation_path/gibson2/global_config.yaml` (default and recommended: `your_installation_path/gibson2/assets`).

Second, you can download our robot models and objects from [here](https://storage.googleapis.com/gibsonassets/assets_gibson_v2.tar.gz) and unpack it in the assets folder.

Third, you need to download some large 3D reconstructed environments (houses, offices) from our dataset for your agents to be trained in. Create a new folder for those environments and set the path in `your_installation_path/gibson2/global_config.yaml` (default and recommended: `your_installation_path/gibson2/assets/dataset`). You can get access and download the full Gibson and iGibson (interactive furniture) datasets by filling up the following [license agreement](https://forms.gle/YTbzXjNtmmsra9KY6). Alternatively, you can download a single [high quality small environment](https://storage.googleapis.com/gibson_scenes/Rs.tar.gz), R's, together with a [fully interactive version](https://storage.googleapis.com/gibson_scenes/Rs_interactive.tar.gz). 

Alternatively, the robot and object models, together with the R's interactive and non-interactive versions can be downloaded and extracted in the assets folder indicated in `your_installation_path/gibson2/global_config.yaml` with a single command:

```bash
python -m gibson2.utis.assets_utils
```

Uninstalling
----

Uninstall iGibson is easy: `pip uninstall gibson2`

