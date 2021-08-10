#!/bin/bash

IMAGE=igibson/igibson

docker build -t $IMAGE:v2.0.0 . \
    && docker tag $IMAGE:v2.0.0 $IMAGE:latest \
    && echo BUILD SUCCESSFUL

# podman build -t $IMAGE:v2.0.0 . \
#     && podman tag $IMAGE:v2.0.0 $IMAGE:latest \
#     && echo BUILD SUCCESSFUL
