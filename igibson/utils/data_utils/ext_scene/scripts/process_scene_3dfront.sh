#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
cd $BASEDIR

THREEDFRONT_DIR=$(python -c "import igibson; print(igibson.threedfront_dataset_path)" | tail -1)
JSON_PATH=$1

python step_1_preprocess_3dfront.py --model_path $JSON_PATH

JSON_PATH=$(basename $JSON_PATH)
THREEDFRONT_ID="${JSON_PATH%.*}"
echo $THREEDFRONT_ID

python step_2_generate_scene.py --overwrite --model_dir \
    $THREEDFRONT_DIR/scenes/$THREEDFRONT_ID
blender -b --python step_3_uv_unwrap.py -- \
    $THREEDFRONT_DIR/scenes/$THREEDFRONT_ID
python step_3_add_mtl.py --input_dir \
    $THREEDFRONT_DIR/scenes/$THREEDFRONT_ID
python step_4_convert_to_ig.py --select_best \
    --source THREEDFRONT $THREEDFRONT_ID
python step_4_convert_to_ig.py --source THREEDFRONT $THREEDFRONT_ID
python step_5_generate_trav_map.py --source THREEDFRONT $THREEDFRONT_ID
