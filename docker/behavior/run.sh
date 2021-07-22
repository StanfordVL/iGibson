#!/usr/bin/env bash

IMAGE=igibson/behavior_challenge_2021

# For docker
docker run --gpus all -ti --rm $IMAGE:latest

# Or if your cluster uses podman
# podman run --rm -it --net=host \
# --security-opt=no-new-privileges \
# --security-opt label=type:nvidia_container_t \
# -e DISPLAY \
# $IMAGE:latest
