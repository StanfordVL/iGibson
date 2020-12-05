# Import an scene into iGibson

To use external scenes in iGibson, it needs be converted to the scene format described here: [data format](../README.md).

Here we provide an automated pipeline to execute this conversion from:
1. [CubiCasa5k](https://github.com/CubiCasa/CubiCasa5k): A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis. (Kalervo, Ahti, et al.)
2. [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset): 3D Furnished Rooms with layOuts and semaNTics. (Fu, Huanl, et al.)
by running a provided script. This script makes use of the software Blender, which needs to be installed first. You won't need to use blender directly, as the script calls it automatically.

The example script executes the following steps
- Preprocess scene data from CubiCasa5k or 3D-FRONT into scene components.
- Generate the visual and collision meshes of the walls, floors and ceiling of the scene. Generate the bounding box annotations of the objects in scene. Generate the base scene URDFs.
- Generate the iGSDF (iGibson Scene Definition Format), an extension of URDF, for the scene. Generate travserability map.

![ignition_toy](images/ext_scenes.jpg)

Above shows some example scenes from CubiCasa5K (first row) and 3D-Front(second row).

We currently only offer support for Linux.

## Installing Blender

We use Blender 2.82 for mesh processing. Follow the instruction here: [blender_utils](../blender_utils/) for guide on installation.

### Process a CubiCasa5K scene

To preprocess a CubiCasa5K scene, please follow these steps:
1. Please follow CubiCasa5K's instruction [here](https://github.com/CubiCasa/CubiCasa5k#dataset) and download the data. After unzipping, the folder (let's call it ```CUBICASA_ROOT```) should contain:
```
colorful  high_quality  high_quality_architectural  test.txt  train.txt  val.txt
``` 

2. Each folder of ```CUBICASA_ROOT/*/* ``` represent the floor plan of a real-world home. We will convert each floor of the home into an iGibson scene, which can be easily done by one command:
```
./scripts/process_scene_cubicasa.sh CUBICASA_ROOT/x/x
```
replace ```CUBICASA_ROOT/x/x``` with the real path.

3. To convert all CubiCasa5K scenes, you can run:
```
for s in CUBICASA_ROOT/*/*; do;  ./scripts/process_scene_cubicasa.sh $s; done
```
replace ```CUBICASA_ROOT``` with the real path. Note that this process can take  multiple hours.

We make the following changes to CubiCasa5K scenes during our processing:
1. We skip objects that are not in our dataset (e.g. Fireplace). We also skip objects that have unclear category label (e.g. GeneralAppliance).
2. For objects that have overlapping bounding boxes, we try to shrink the bounding boxes by no more than 20%. We sort all objects by how many other objects the object overlaps with. We iteratively go through the list, if shrinking doesn't resolve the issue, we delete that object.

Notes:
1. CubiCasa5k are floorplans of fixed furnitures only (e.g. stove, wall cabinet, fridge etc.). 
2. Object and scene category mappings can be found [here](scripts/utils/semantics.py) .

### Process a 3D-FRONT scene

To import a 3D-Front scene into iGibson, please follow the these steps:
1. Please follow 3D-Front's instruction [here](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset#download) and download the data. After unzipping, the folder (let's call it ```THREEDFRONT_ROOT```) should contain 10k *.json* files.
2. Each file in ```THREEDFRONT_ROOT``` represent a online-designed apartment. We will convert each apartment into an iGibson scene, which can be easily done by one command:
```
./scripts/process_scene_3dfront.sh THREEDFRONT_ROOT/x
```
replace ```THREEDFRONT_ROOT/x``` with the real path of any json file.

3. To convert all 3D-Front scenes, you can run:
```
for s in THREEDFRONT_ROOT/*; do;  ./scripts/process_scene_3dfront.sh $s; done
```
replace ```THREEDFRONT_ROOT``` with the real path. Note that this process can take multiple hours.

We make the following changes to 3D-Front scenes during our processing:
1. We skip scenes that contain unclear category label as part of the scene mesh:
```
[ 'CustomizedFixedFurniture', 'CustomizedFurniture', 'CustomizedPersonalizedModel', 'CustomizedPlatform']
```
since we can't generate collision meshes properly with these categories. In total, there are 8808 scenes that don't contain any of the category above.
2. Real-world free-moving objects can have overlapping bounding boxes (e.g. a chair tucked into a table), thus, for overlapping bounding box issue, we do the following:
- if an object bounding box is over 80% contained in another object, this is likely to be a scene design overlook (for example, penetrating furniture in the original scene design, see [here](https://github.com/3D-FRONT-FUTURE/3D-FRONT-ToolBox/issues/4) ). We thus skip the object.
- if two objects overlap with each other, we try to shrink the bounding boxes. We shrink by no more than 80%.
- we randomize the objects with our object assets, and manage to provide non-overlapping layouts for 2200 scenes. You can download the URDFs via:
```
python -m gibson2.utils.assets_utils --download_threedfront_nonoverlapping_urdfs
```
3. 3D-Front also has known corrupted mesh issue (see [github issue](https://github.com/3D-FRONT-FUTURE/3D-FRONT-ToolBox/issues/2#issuecomment-682678930)). 
 
Notes:
1.  Object and scene category mappings can be found [here](scripts/utils/semantics.py) .
