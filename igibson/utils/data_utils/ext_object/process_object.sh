#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
echo "$BASEDIR"
cd $BASEDIR
#########
# The script takes inputs:
# 1. directory to the source files
# 2. category label of the object
#########
IGIBSON_DIR=$(python -c "import igibson; print(igibson.ig_dataset_path)" | tail -1)
DIRECTORY=$1
OBJECT_ID=$(basename "$1")
CATEGORY=$2
OBJECT_EXPORT_DIR=$IGIBSON_DIR/objects/$CATEGORY/$OBJECT_ID

cd scripts
##################
# Generate visual meshes 
##################
blender -b --python step_1_visual_mesh.py -- --source_dir $DIRECTORY --dest_dir $OBJECT_EXPORT_DIR #--up Z --forward X 

##################
# Generate collision meshes
##################
python step_2_collision_mesh.py \
    --input_dir $OBJECT_EXPORT_DIR/shape/visual \
    --output_dir $OBJECT_EXPORT_DIR/shape/collision \
    --object_name $OBJECT_ID --split_loose

##################
# Generate misc/*.json
##################
python step_3_metadata.py --input_dir $OBJECT_EXPORT_DIR

##################
# Generate .urdf
##################
python step_4_urdf.py --input_dir $OBJECT_EXPORT_DIR

##################
# Generate visualizations
##################
python step_5_visualizations.py --input_dir $OBJECT_EXPORT_DIR


