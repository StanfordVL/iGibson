#!/bin/bash
# This file should be run on an off-cluster machine such as capri.

IMAGE=igibson-dev

docker build -t $IMAGE . || {
    echo 'Could not build image.' ;
    exit 1;
}

enroot import --output /cvgl/group/igibson-docker/igibson-dev.sqsh dockerd://${IMAGE} || {
    echo 'Could not import image.' ;
    exit 1;
}

docker rmi