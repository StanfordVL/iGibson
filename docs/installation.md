
Installation
=================

There are two steps to get iGibson, the Interactive Gibson Environment, on your computer. First, you need to install the simulation environment. Then, you need to download some 3D reconstructed large environments for your agents to train.

#### Installing the Environment

There are two methods to install iGiboson.

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

#### Download Dataset of 3D Environments

Our environment core assets data are available [here](https://storage.googleapis.com/gibsonassets/assets_gibson_v2.tar.gz).  You can set the path in `global_config.yaml`.  The `assets` folder stores necessary data (agent models, environments, etc) to run gibson environment. Assets data can be downloaded and extracted with a single command:

```bash
python -m gibson2.utis.assets_utils
```

Users can add more environments files into `dataset` folder and put the path in `global_config.yaml` to run gibson on more environments. Visit the [database readme](https://forms.gle/YTbzXjNtmmsra9KY6) for downloading more spaces. Please sign the [license agreement](https://forms.gle/YTbzXjNtmmsra9KY6) before using Gibson's database. The default path is:

```yaml
assets_path: assets #put either absolute path or relative to current directory
dataset_path: assets/dataset
```

#### Download Full Interactive Scene

We include

By running:
```bash
python -m gibson2.utis.assets_utils
```

You will download `Rs` scene and `Rs_interactive` into `dataset_path` and you should be able to run all the tests and some examples. Full dataset will can be downloaded [here](https://forms.gle/YTbzXjNtmmsra9KY6).


Uninstalling
----

Uninstall gibson is easy with `pip uninstall gibson2`

