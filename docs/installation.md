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



### (Optional) Create a docker image for iGibson


If you want to install gibson for cross-platform use and don't want to set up the dependencies. You can use the
following Dockerfile to create a docker image for using iGibson.  `nvidia-docker` is required to run this docker image.

```text
from nvidia/cudagl:10.0-base-ubuntu18.04

ARG CUDA=10.0
ARG CUDNN=7.6.2.24-1

RUN apt-get update && apt-get install -y --no-install-recommends \
	curl build-essential git cmake \
	cuda-command-line-tools-10-0 \
    cuda-cublas-10-0 \
    cuda-cufft-10-0 \
    cuda-curand-10-0 \
    cuda-cusolver-10-0 \
    cuda-cusparse-10-0 \
    libcudnn7=${CUDNN}+cuda${CUDA} \
    vim \
    tmux \
    libhdf5-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda create -y -n py3-igibson python=3.6.8
# Python packages from conda

ENV PATH /miniconda/envs/py3-igibson/bin:$PATH

RUN pip install pytest
RUN pip install tensorflow-gpu==1.15.0

RUN git clone --branch master https://github.com/StanfordVL/iGibson /opt/igibson --recursive
WORKDIR /opt/igibson
RUN pip install -e .

RUN git clone https://github.com/StanfordVL/agents/ /opt/agents
WORKDIR /opt/agents
RUN pip install -e .

RUN python -m gibson2.utils.assets_utils --download_assets
RUN python -m gibson2.utils.assets_utils --download_demo_data

```


### Uninstalling
Uninstalling iGibson is easy: `pip uninstall gibson2`

