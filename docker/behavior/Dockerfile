FROM nvidia/cudagl:11.1.1-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
	cmake \
	curl \
	g++ \
	git \
	make \
	vim \
	wget \
	cuda-command-line-tools-11-1 && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda create -y -n igibson python=3.8.0

ENV PATH /miniconda/envs/igibson/bin:$PATH

RUN pip install --no-cache-dir torch pytest ray[default,rllib] stable-baselines3 && rm -rf /root/.cache

RUN git clone --depth 1 --branch ig-develop https://github.com/StanfordVL/iGibson /opt/iGibson --recursive && \
        rm -rf /opt/iGibson/igibson/render/openvr/samples

WORKDIR /opt/iGibson

RUN pip install --no-cache-dir -e .

RUN git clone --depth 1 --branch master https://github.com/StanfordVL/BDDL /opt/BDDL --recursive
WORKDIR /opt/BDDL
RUN pip install --no-cache-dir -e .

RUN python3 -m igibson.utils.assets_utils --download_assets && rm -rf /tmp
RUN python3 -m igibson.utils.assets_utils --download_demo_data && rm -rf /tmp

WORKDIR /opt/iGibson/igibson
