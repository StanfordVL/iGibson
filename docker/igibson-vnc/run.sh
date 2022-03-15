#!/usr/bin/env bash

VNC_PORT=5900
VNC_PASSWORD=112358

PARAMS=
while (( "$#" )); do
  case "$1" in
    -p|--vnc-port)
      VNC_PORT=$2
      shift 2
      ;;
    -pw|--vnc-password)
      VNC_PASSWORD=$2
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done

echo Starting VNC server on port $VNC_PORT with password $VNC_PASSWORD
echo please run \"python -m igibson.examples.environments.env_nonint_example\" once you see the docker command prompt:
docker run --gpus all -ti -p $VNC_PORT:5900 -e VNC_PASSWORD=$VNC_PASSWORD --rm igibson/igibson-vnc:latest bash
