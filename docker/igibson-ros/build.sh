#!/bin/bash

IMAGE=igibson/igibson-ros

docker build -t $IMAGE .

# podman build -t $IMAGE .
