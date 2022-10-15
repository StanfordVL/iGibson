#!/usr/bin/env bash

enroot create -n igibson-${SLURM_LOCALID} /cvgl/group/igibson-docker/igibson-dev.sqsh && {
  # Run the container, mounting iGibson at the right spot
  enroot start -r -w \
    -m ${IG_IGIBSON_PATH}:/igibson \
    -m ${IG_OUTPUT_PATH}:/out \
    -e SLURM_LOCALID=${SLURM_LOCALID} \
    -e IG_ENTRYPOINT_MODULE=${IG_ENTRYPOINT_MODULE} \
    igibson-${SLURM_LOCALID};
  # Remove the container.
  enroot remove -f igibson-${SLURM_LOCALID};
}