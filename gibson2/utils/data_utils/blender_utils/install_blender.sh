#!/usr/bin/env bash

curl -O https://download.blender.org/release/Blender2.82/blender-2.82-linux64.tar.xz
tar -xvf blender-2.82-linux64.tar.xz
echo "export PATH=\$PATH:$(pwd)/blender-2.82-linux64" >> ~/.bashrc
