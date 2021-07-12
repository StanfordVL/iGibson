#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
cd $BASEDIR

CUBICASA_DIR=$(python -c "import igibson; print(igibson.cubicasa_dataset_path)" | tail -1)
DIRECTORY=$1

python step_1_preprocess_cubicasa.py --model_dir $1

CUBICASA_ID=$(basename $DIRECTORY)
echo $CUBICASA_ID

for scene in $CUBICASA_DIR/scenes/"$CUBICASA_ID"_*; do
    echo $scene
    python step_2_generate_scene.py --model_dir $scene
    blender -b --python step_3_uv_unwrap.py $scene
    python step_3_add_mtl.py --input_dir $scene
    python step_4_convert_to_ig.py --select_best \
        --source CUBICASA $(basename $scene) 
    python step_4_convert_to_ig.py --source CUBICASA $(basename $scene) 
    python step_5_generate_trav_map.py --source CUBICASA $(basename $scene) 
done
