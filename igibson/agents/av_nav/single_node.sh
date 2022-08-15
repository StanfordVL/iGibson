#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export MASTER_PORT=8080

# python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'

set -x
# FREEPORT = $(python find_free_port.py 2>&1)
python -m torch.distributed.launch --nproc_per_node 2 run.py --exp-config config/audiopointgoal_single.yaml --free_port $(python find_free_port.py 2>&1)