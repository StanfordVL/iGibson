#!/usr/bin/env bash

CONTAINER_NAME=igibson-${SLURM_JOBID}-${SLURM_ARRAY_TASK_ID}-${SLURM_LOCALID}

enroot create -n ${CONTAINER_NAME} /cvgl/group/igibson-docker/igibson-data.sqsh && {
  # Run the container, mounting iGibson at the right spot
  enroot start -r -w \
    -m ${IG_DATA_PATH}:/opt/igibson/igibson/data \
    -m ${IG_OUTPUT_PATH}:/out \
    -e SLURM_JOBID=${SLURM_JOBID} \
    -e SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} \
    -e SLURM_LOCALID=${SLURM_LOCALID} \
    ${CONTAINER_NAME} \
    ${IG_ENTRYPOINT_COMMAND};
  # Remove the container.
  enroot remove -f ${CONTAINER_NAME};
}