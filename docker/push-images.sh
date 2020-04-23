#!/usr/bin/env bash

VERSION=0.0.4

docker push igibson/igibson:v$VERSION
docker push igibson/igibson:latest
docker push igibson/igibson-gui:v$VERSION
docker push igibson/igibson-gui:latest
