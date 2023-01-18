#!/usr/bin/env bash

# For docker
docker run --gpus all -ti --rm igibson/igibson-ros:latest

# Or if your cluster uses podman
# podman run --rm -it --net=host \
# --security-opt=no-new-privileges \
# --security-opt label=type:nvidia_container_t \
# -e DISPLAY \
# igibson/igibson:latest
