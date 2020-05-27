#!/bin/bash

IMAGE=igibson/igibson

export DOCKER_BUILDKIT=1
docker build -t $IMAGE:v0.0.4 . \
    && docker tag $IMAGE:v0.0.4 $IMAGE:latest \
    && echo BUILD SUCCESSFUL
