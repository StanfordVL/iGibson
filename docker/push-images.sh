#!/usr/bin/env bash

VERSION=1.0.1

docker push igibson/igibson:v$VERSION
docker push igibson/igibson:latest
docker push igibson/igibson-gui:v$VERSION
docker push igibson/igibson-gui:latest
