#!/usr/bin/env bash

VERSION=0.0.4

docker pull igibson/igibson:v$VERSION
docker pull igibson/igibson:latest
docker pull igibson/igibson-gui:v$VERSION
docker pull igibson/igibson-gui:latest
