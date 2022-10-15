#!/bin/bash
# This file should be run on an off-cluster machine such as capri.

IMAGE=igibson-dev
OUTPUT_PATH=/cvgl/group/igibson-docker/${IMAGE}.sqsh

echo "OUTPUT_PATH="${OUTPUT_PATH};

{
if [ -f OUTPUT_PATH ]; then
    echo "Output file ${OUTPUT_PATH} already exists";
    exit 1;
fi
}

docker build -t $IMAGE . || {
    echo 'Could not build image.' ;
    exit 1;
}

enroot import --output  dockerd://${IMAGE} || {
    echo 'Could not import image.' ;
    exit 1;
}

docker rmi $IMAGE
