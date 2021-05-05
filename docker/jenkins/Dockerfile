from nvidia/cudagl:10.0-base-ubuntu18.04

ARG CUDA=10.0
ARG CUDNN=7.6.2.24-1

RUN apt-get update && \
    apt-get -y install sudo

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID jenkins
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID jenkins 

ENV user jenkins

RUN chown -R ${user} /home/${user} && \
    adduser ${user} sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers


USER ${user}
WORKDIR /home/${user}

RUN sudo apt-get install -y --no-install-recommends \
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
RUN bash Miniconda-latest-Linux-x86_64.sh -p /home/${user}/miniconda -b
RUN rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/home/${user}/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda create -y -n py3-igibson python=3.6.8
# Python packages from conda

ENV PATH /home/${user}/miniconda/envs/py3-igibson/bin:$PATH

RUN pip install pytest
RUN pip install Cython==0.21.1