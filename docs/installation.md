# Installation
There are two steps to install iGibson, the Interactive Gibson Environment, on your computer.

First, you need to install the simulation environment. Then, you need to download the assets: models of the robotic agents, the interactive objects and 3D reconstructed real-world large environments for your agents to train.

### Installing the Environment

We provide two methods to install the simulator.

First, iGibson's simulator can be installed as a python package using pip:

```bash
pip install gibson2
# run the demo
python -m gibson2.envs.demo
python -m gibson2.envs.demo_interactive
```

Alternatively, it can be compiled from source: [iGibson GitHub Repo](https://github.com/StanfordVL/iGibson)

```bash
git clone https://github.com/StanfordVL/iGibson --recursive
cd iGibson

conda create -n py3-igibson python=3.6 anaconda
source activate py3-igibson
pip install -e .
```
We recommend the second method if you plan to modify iGibson in your project. If you plan to use it as it is to train navigation and manipulation agents, the pip installation should meet your requirements.

#### System Requirements

The minimum system requirements are the following:

- Ubuntu 16.04
- Nvidia GPU with VRAM > 6.0GB
- Nvidia driver >= 384
- CUDA >= 9.0, CuDNN >= v7

Other system configurations may work, but we haven't tested them extensively and we probably won't be able to provide as much support as we want.

### Downloading the Assets
TODO: @fei
1. Change default value of global_config.yaml to be gibson2/assets and gibson2/dataset
2. Users should always use assets_utils to download assets and Rs and Rs_interactive
3. Users need to fill up the form to download Gibson houses: a) 10 interactive ones, b) 572 static ones
4. Put them into gibson2/dataset

First, create a folder to contain all the iGibson's assets (robotic agents, objects, 3D environments, etc.) and set the path in `your_installation_path/gibson2/global_config.yaml` (default and recommended: `your_installation_path/gibson2/assets`).

Second, you can download our robot models and objects from [here](https://storage.googleapis.com/gibson_scenes/assets_igibson.tar.gz) and unpack it in the assets folder.

Third, you need to download some large 3D reconstructed real-world environments (houses, offices) from our dataset for your agents to be trained in. Create a new folder for those environments and set the path in `your_installation_path/gibson2/global_config.yaml` (default and recommended: `your_installation_path/gibson2/assets/dataset`). You can get access and download the full Gibson and iGibson (interactive furniture) datasets by filling up the following [license agreement](https://forms.gle/YTbzXjNtmmsra9KY6). Alternatively, you can download a single [high quality small environment](https://storage.googleapis.com/gibson_scenes/Rs.tar.gz), R's, together with a [fully interactive version](https://storage.googleapis.com/gibson_scenes/Rs_interactive.tar.gz). 

Alternatively, the robot and object models, together with the R's interactive and non-interactive versions can be downloaded and extracted in the assets folder indicated in `your_installation_path/gibson2/global_config.yaml` with a single command:

```bash
python -m gibson2.utils.assets_utils
```

### Uninstalling
Uninstalling iGibson is easy: `pip uninstall gibson2`

