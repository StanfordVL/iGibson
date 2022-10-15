FROM nvidia/cudagl:11.3.1-base-ubuntu20.04

ADD entrypoint.sh /root/entrypoint.sh

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
	cmake \
	curl \
	g++ \
	git \
	make \
	vim \
	wget \
	cuda-command-line-tools-11-3 && \
    rm -rf /var/lib/apt/lists/*

# Needed for QT window for cv2
RUN apt-get update && apt-get install -y --no-install-recommends \
	libsm6 && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}

RUN conda update -y conda
RUN conda create -y -n igibson python=3.8

ENV PATH /miniconda/envs/igibson/bin:$PATH

# NOTE: This needs to be updated in-step with the base cudagl image so the tensor renderer works
RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html --no-cache-dir

ENTRYPOINT /igibson/entrypoint.sh
