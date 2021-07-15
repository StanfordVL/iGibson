#!/bin/bash

BASEDIR=$(dirname "$0")
echo "$BASEDIR"
cd $BASEDIR
#########
# The script takes inputs:
# 1. directory to the source files
# 2. category label of the object
#########
IGIBSON_DIR=$(python -c "import igibson; print(igibson.ig_dataset_path)" | tail -1)
SRC_DIR=$1
OUTOUT_DIR=$2
CATEGORY=$3
OBJECT_ID=$4
# OBJECT_EXPORT_DIR=$IGIBSON_DIR/objects/$CATEGORY/$NAME/$OBJECT_ID
OBJECT_EXPORT_DIR=$OUTOUT_DIR/$CATEGORY/$OBJECT_ID

echo $OBJECT_EXPORT_DIR

echo "Step 1"
cd scripts
##################
# Generate visual meshes
##################
blender -b --python step_1_visual_mesh_multi_uv.py -- --bake --up Y --forward -Z --source_dir $SRC_DIR --dest_dir $OBJECT_EXPORT_DIR --merge_obj 0

echo "Step 2"
##################
# Generate collision meshes
##################
python step_2_collision_mesh.py \
    --input_dir $OBJECT_EXPORT_DIR/shape/visual \
    --output_dir $OBJECT_EXPORT_DIR/shape/collision \
    --object_name $OBJECT_ID \
    --urdf $SRC_DIR/../mobility.urdf

echo "Step 3"
##################
# Generate misc/*.json
##################
python step_3_metadata.py --input_dir $OBJECT_EXPORT_DIR

echo "Step 4"
##################
# Generate .urdf
##################
python step_4_urdf.py --input_dir $OBJECT_EXPORT_DIR --urdf $SRC_DIR/../mobility.urdf

echo "Step 5"
##################
# Generate visualizations
##################
python step_5_visualizations.py --input_dir $OBJECT_EXPORT_DIR


