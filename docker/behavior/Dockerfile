FROM nvidia/cudagl:11.1.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
	cmake \
	git \
	vim \
	wget \
	curl \
	python3-dev \
	python3-opencv \
	python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda create -y -n igibson python=3.8.0

ENV PATH /miniconda/envs/igibson/bin:$PATH

RUN pip install torch pytest ray[default,rllib] stable-baselines3

RUN git clone --branch behavior_docker https://github.com/StanfordVL/iGibson /opt/iGibson --recursive
WORKDIR /opt/iGibson
RUN pip install -e .

RUN git clone --branch master https://github.com/StanfordVL/BDDL /opt/BDDL --recursive
WORKDIR /opt/BDDL
RUN pip install -e .

RUN python3 -m igibson.utils.assets_utils --download_assets
RUN python3 -m igibson.utils.assets_utils --download_demo_data

WORKDIR /opt/iGibson/igibson
