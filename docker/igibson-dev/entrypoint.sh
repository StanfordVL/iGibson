#! /bin/bash
# This file should be run within the container, at runtime.

cd /igibson || {
    echo 'Could not cd into igibson dir.' ;
    exit 1;
}

pip install --no-cache-dir -e . || {
    echo 'Could not install igibson.' ;
    exit 1;
}

rm -rf /root/.cache

python -m ${IG_ENTRYPOINT_MODULE}