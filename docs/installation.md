
Installation
=================

#### Installation Method

InteractiveGibsonEnv can be installed as a python package:

```bash
git clone https://github.com/StanfordVL/InteractiveGibsonEnv --recursive
cd InteractiveGibsonEnv

conda create -n py3-gibson python=3.6 anaconda
source activate py3-gibson
pip install -e .
```

#### System requirements

The minimum system requirements are the following:

- Ubuntu 16.04
- Nvidia GPU with VRAM > 6.0GB
- Nvidia driver >= 384
- CUDA >= 9.0, CuDNN >= v7

#### Download data

First, our environment core assets data are available [here](https://storage.googleapis.com/gibsonassets/assets_gibson_v2.tar.gz).  You can store the data where you want and put the path in `global_config.yaml`.  The `assets` folder stores necessary data (agent models, environments, etc) to run gibson environment. 

Users can add more environments files into `dataset` folder and put the path in `global_config.yaml` to run gibson on more environments. Visit the [database readme](https://forms.gle/YTbzXjNtmmsra9KY6) for downloading more spaces. Please sign the [license agreement](https://forms.gle/YTbzXjNtmmsra9KY6) before using Gibson's database. The default path is:

```yaml
assets_path: assets #put either absolute path or relative to current directory
dataset_path: assets/dataset
```

#### Download sample scenes

```
wget https://storage.googleapis.com/gibsonassets/gibson_mesh/Ohopee.tar.gz
```
Put the downloaded `Ohopee` scene into `dataset_path` and you should be able to run all the tests and some examples. Full dataset will can be downloaded [here](https://forms.gle/YTbzXjNtmmsra9KY6).


Uninstalling
----

Uninstall gibson is easy with `pip uninstall gibson2`

