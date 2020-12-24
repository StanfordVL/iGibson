#!/bin/bash

IMAGE=igibson/igibson

export DOCKER_BUILDKIT=1
docker build -t $IMAGE:v1.0.1 . \
    && docker tag $IMAGE:v1.0.1 $IMAGE:latest \
    && echo BUILD SUCCESSFUL
