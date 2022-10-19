#! /bin/bash
# This file should be run within the container, at runtime.

cd /igibson || {
    echo 'Could not cd into igibson dir.' ;
    exit 1;
}

export SETUPTOOLS_ENABLE_FEATURES="legacy-editable";

pip install --no-cache-dir -e . || {
    echo 'Could not install igibson.' ;
    exit 1;
}

eval ${IG_ENTRYPOINT_COMMAND}