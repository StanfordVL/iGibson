# Import an object into iGibson

We provide instructions and scripts here to include your own model (rigid body without articulation) to iGibson. 

We currently only offer support for Linux.

## Installing Blender

We use Blender 2.82 for mesh processing. Follow the instruction here: [blender_utils](../blender_utils/) for guide on installation.

## Prerequisites on object data

To import a rigid-body object, the following files and properties are needed as prerequisite:
1. all relevant files (meshes, textures) are in the same folder 
2. **Mesh(es)**: the object can consist of a single *.obj* file or a list of *.obj* files. The materials (*.mtl*, and textures files) are correctly linked. All *.obj* should be in the same directory, and the directory should contain *.obj*s only from the given object. 
3. **Pose**: meshes are correctly facing forward (e.g. a camera's lens is front-facing)
4. **Scale**: meshes have correct scale.
5. **Category**: the label (string) of the object is available.
Note: You can make sure the meshes are correct by importing the meshes into MeshLab/Blender.

## Default end-to-end processing of the data

Our processing on the data consists of various components each of which can be customized. We provide a simple script that uses default options for all steps. 

To process an object with default options, two parameters need to be specified:
1. folder **$DIRECTORY** in which the files live.
2. cateogry **$CATEGORY** label of the object.

Additionally, the meshes are assumed to have **positive-Z axis pointing up** and **positive positive-X axis pointing forward**. (if not, you can pass in the correct axis in [Step 1](#step-1-visual-mesh-processing))

You can run the following command:
```
./process_object.sh $DIRECTORY $CATEGORY
```

The script will perform the following operations:
- Generate the object at location *$(gibson2.ig_dataset_path)/objects/$CATEGORY/$(os.path.basename($DIRECTORY))*.
- [Step 1](#step-1-visual-mesh-processing): Process the original meshes using Blender and export as visual meshes to *shape/visual*.
- [Step 2](#step-2-collision-mesh-processing): For all meshes, calculate the collision mesh in *shape/collision* using [V-HACD](https://github.com/kmammou/v-hacd) and optionally Blender.
- [Step 3](#step-3-generating-object-link-data): Generate *misc/metadata.json, misc/material_groups.json*.
- [Step 4](#step-4-generating-object-urdf): Generate object's URDF file.
- [Step 5](#step-5-generating-visualization): Generate video visualization of object rendered in iGibson renderer.

Each step will use the default options.

### Step 1: visual mesh processing

For step 1, the following script is used: [scripts/step_1_visual_mesh.py](scripts/step_1_visual_mesh.py). 
The script performs the following functionalities:
1. core functionality: import the *.obj* files to Blender. And
- write out meshes in Z-up X-forward format; 
- write out vertex normal;
- write out all faces as triangles.
2. optional functionality:
- [texture baking](https://docs.blender.org/manual/en/latest/render/cycles/baking.html) (by passing in flag --bake): baking textures for rendering in iGibson engines. The baking process will first UV-unwrap the object mesh. Then four channels used for iGibson's PBR rendering will be baked: diffuse color, tangent normal, metallic, roughness.
**Note**: if your object model contains multiple texture maps (aka *.mtl* file links to image textures), please **do not** perform baking.

Required parameters:
1. --source_dir: the source directory where the object assets are located.
2. --dest_dir: the destination directory where the object assets should be exported to. To use in iGibson, the path should be: *$IGIBSON_ROOT/objects/$CATEGORY/$OBJECT_NAME*.

Additional flags of the script are:
1. --bake: whether textures should be baked. See above.
2. --up: up axis of the original assets, should be among {X,Y,Z,-X,-Y,-Z}.
3. --forward: forward axis of the original assets, should be among {X,Y,Z,-X,-Y,-Z}.

Example use:
```
blender -b --python step_1_visual_mesh.py --bake --up Z --forward X --source_dir {PATH_TO_YOUR_OBJECT} --dest_dir {PATH_TO_IGIBSON_ASSET}/objects/{OBJECT_CATEGORY}/{OBJECT_NAME}
```

### Step 2: collision mesh processing

For step 2, the following script is used: [scripts/step_2_collision_mesh.py](scripts/step_2_collision_mesh.py). 
The script performs the following functionalities:
1. core funtionality: for each visual mesh component, calculate its collision mesh using [V-HACD](https://github.com/kmammou/v-hacd), and merge all the generated collision meshes into a final collision mesh for the object.
2. optional functionality: 
- splitting loose parts (by passing in flag --split_loose): when calculating the collision mesh, first split each visual mesh into its loose parts, and calculate collision mesh for each part. Note that this will create a more fine-grained collision mesh, but can induce a heavier computation load when performing physics simulation in iGibson. 

Required parameters:
1. --input_dir: directory containing the visual meshes  (by iGibson data format, it should be in *shape/visual*).
2. --output_dir: directory in which the collision mesh should be stored (by iGibson data format, it should be in *shape/collision*).
3. --object_name: name of the object.

Example use:
```
python step_2_collision_mesh.py --input_dir {PATH_TO_IGIBSON_ASSET}/objects/{OBJECT_CATEGORY}/{OBJECT_NAME}/shape/visual --output_dir {PATH_TO_IGIBSON_ASSET}/objects/{OBJECT_CATEGORY}/{OBJECT_NAME}/shape/collision --object_name {OBJECT_NAME} --split_loose
```


### Step 3: generating object meta-data 

For step 3, the following script is used: [scripts/step_3_metadata.py](scripts/step_3_metadata.py). 
The script performs the following functionalities:
1. Calculate the bounding box information of the object, which is used in iGibson's object randomization.
2. Generate the material information of the object, which is used in iGibson's material randomization.

To specify the material of your object, you can use --material to pass in a list of comma separated materials (e.g. *--material wood,metal,marble*). The supported materials of iGibson are:
```
['asphalt', 'bricks', 'ceramic', 'metal_diamond_plate', 'paper', 'chipboard', 'wood_floor', 'rocks', 'ground', 'fabric', 'snow', 'plastic', 'rubber', 'paint', 'wood', 'fabric_carpet', 'porcelain', 'tiles', 'corrugated_steel', 'plaster', 'moss', 'terrazzo', 'paving_stones', 'leather', 'metal', 'marble', 'concrete', 'planks']
```
Required parameters:
1. --input_dir: the root directory of the object, which should be: *$IGIBSON_ROOT/objects/$CATEGORY/$OBJECT_NAME*.

Example use:
```
python step_3_metadata.py --input_dir {PATH_TO_IGIBSON_ASSET}/objects/{OBJECT_CATEGORY}/{OBJECT_NAME} --material ceramic,marble
```

### Step 4: generating object URDF

For step 4, the following script is used: [scripts/step_4_urdf.py](scripts/step_4_urdf.py). 
The script generates a simple single-link URDF for the object.

Required parameters:
1. --input_dir: the root directory of the object, which should be: *$IGIBSON_ROOT/objects/$CATEGORY/$OBJECT_NAME*.

Optional parameters:
1. --mass: the mass of the object.

Example use:
```
python step_4_urdf.py --input_dir {PATH_TO_IGIBSON_ASSET}/objects/{OBJECT_CATEGORY}/{OBJECT_NAME} --mass 100
```

### Step 5: generating visualization
 
For step 5, the following script is used: [scripts/step_5_visualizations.py](scripts/step_5_visualizations.py). 
The script generates a video visualization of the object rendered in iGibson.

Required parameters:
1. --input_dir: the root directory of the object, which should be: *$IGIBSON_ROOT/objects/$CATEGORY/$OBJECT_NAME*.

Example use:
```
python step_5_visualizations.py --input_dir {PATH_TO_IGIBSON_ASSET}/objects/{OBJECT_CATEGORY}/{OBJECT_NAME}
```
