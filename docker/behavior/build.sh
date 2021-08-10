#!/bin/bash

IMAGE=igibson/behavior_challenge_2021

docker build -t $IMAGE . \
    && echo BUILD SUCCESSFUL

# podman build -t $IMAGE . \
#     && echo BUILD SUCCESSFUL
