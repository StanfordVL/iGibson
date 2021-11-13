#!/bin/bash

IMAGE=igibson/igibson
VERSION=v2.0.3

podman build -t $IMAGE:$VERSION . \
    && podman tag $IMAGE:$VERSION $IMAGE:latest \
    && echo BUILD SUCCESSFUL

# podman build -t $IMAGE:v2.0.0 . \
#     && podman tag $IMAGE:v2.0.0 $IMAGE:latest \
#     && echo BUILD SUCCESSFUL
