FROM nvidia/cudagl:11.3.1-base-ubuntu20.04

# install ros-core and ros-base

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO noetic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-core=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*
    
# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*
    

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

RUN git clone --depth 1 https://github.com/StanfordVL/iGibson /opt/iGibson --recursive

WORKDIR /opt/iGibson
RUN pip install --no-cache-dir -e .

RUN pip install --no-cache-dir pytest ray[default,rllib] stable-baselines3 && rm -rf /root/.cache


RUN python3 -m igibson.utils.assets_utils --download_assets && rm -rf /tmp/*
RUN python3 -m igibson.utils.assets_utils --download_demo_data && rm -rf /tmp/*

WORKDIR /opt/iGibson/igibson/examples

RUN  apt-get update && apt-get install -y mesa-utils

RUN mkdir -p /opt/catkin_ws/src
RUN ln -s /opt/iGibson/igibson/examples/ros/igibson-ros /opt/catkin_ws/src/igibson-ros

ENV PYTHONPATH /opt/ros/noetic/lib/python3/dist-packages:/miniconda/envs/igibson/lib/python3.8/site-packages/:/usr/lib/python3/dist-packages:/opt/iGibson/

RUN cd /opt/catkin_ws && /opt/ros/noetic/env.sh catkin_make


RUN cd /opt/catkin_ws && /opt/ros/noetic/env.sh rosdep install --from-paths src --ignore-src -r -y

WORKDIR /opt/catkin_ws/src/igibson-ros

# run source /opt/catkin_ws/devel/setup.bash after getting into docker