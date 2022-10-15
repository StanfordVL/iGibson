#!/usr/bin/env bash

CONTAINER_NAME=igibson-${SLURM_ARRAY_TASK_ID}-${SLURM_LOCALID}

enroot create -n ${CONTAINER_NAME} /cvgl/group/igibson-docker/igibson-dev.sqsh && {
  # Run the container, mounting iGibson at the right spot
  enroot start -r -w \
    -m ${IG_IGIBSON_PATH}:/igibson \
    -m ${IG_OUTPUT_PATH}:/out \
    -e SLURM_JOBID=${SLURM_JOBID} \
    -e SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} \
    -e SLURM_LOCALID=${SLURM_LOCALID} \
    -e IG_ENTRYPOINT_MODULE=${IG_ENTRYPOINT_MODULE} \
    ${CONTAINER_NAME};
  # Remove the container.
  enroot remove -f ${CONTAINER_NAME};
}