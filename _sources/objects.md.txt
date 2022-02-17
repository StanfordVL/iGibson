# Objects

### Overview
We provide a wide variety of **Objects** that can be imported into the **Simulator**.
- `YCBObject`
- `RBOObject`
- `ShapeNetObject`
- `Pedestrian`
- `ArticulatedObject`
- `URDFObject`
- `SoftObject`
- `Cube`
- `VisualMarker`

Typically, they take in the name or the path of an object (in `igibson.assets_path`) and provide a `load` function that be invoked externally (usually by `import_object` and `import_object` of `Simulator`). The `load` function imports the object into PyBullet. Some **Objects** (e.g. `ArticulatedObject`) also provide APIs to get and set the object pose.

Most of the code can be found here: [igibson/objects](https://github.com/StanfordVL/iGibson/blob/master/igibson/objects).

### BEHAVIOR Dataset of Objects

We use a new dataset of everyday objects, the BEHAVIOR Dataset of Objects. In total, we curate 1217 object models across 391 object categories, to support 100 BEHAVIOR activities. The categories range from food items to tableware, from home decorations to office supplies, and from apparel to cleaning tools.

To maintain high visual realism, all object models include material information (metallic, roughness) that can be rendered by iGibson 2.0 renderer. To maintain high physics realism, object models are annotated with size, mass, center of mass, moment of inertia, and also stable orientations. The collision mesh is a simplified version of the visual mesh, obtained with a convex decomposition using the VHACD algorithm. Object models with a shape close to a box are annotated with a primitive box collision mesh, much more efficient and robust for collision checking.

All models in the BEHAVIOR Dataset are organized following the WordNet, associating them to synsets. This structure allows us to define properties for all models of the same categories, but it also facilitates more general sampling of activity instances fulfilling initial conditions such as onTop(fruit, table) that can be achieved using any model within the branch fruit of WordNet.

Please see below for an updated file structure for BEHAVIOR Dataset

```bash
OBJECT_NAME
│   # Unified Robot Description Format (URDF)
│   # http://wiki.ros.org/urdf
│   # It defines the object model (parts, articulation, dynamics properties etc.).
│   OBJECT_NAME.urdf 
│
└───shape
│   └───visual 
│   │    │   # Directory containing visual meshes (vm) of the object. Used for iGibson's rendering. Encrypted
│   │    │   # All objs are UV mapped onto the same texture, linked by default.mtl. All faces are triangles.
│   │    │  vm1.encrypted.obj
│   │    │  vm2.encrypted.obj
│   │    │  …
│   │    │  default.mtl (links the geometry to the texture files)
│   │ 
│   └───collision
│   │    │   # Directory containing collision meshes (cm) of the objects. Used for iGibson's physics simulation.
│   │    │   # Each obj represents a unique link of the object.
│   │    │   # For example, link_1_cm.obj represents the collision mesh of link_1. 
│   │    │  cm1.obj
│   │    │  cm2.obj
│   │    │  …
│
└───material
│   │   # Contains 4 default channels:
│   │   # 	DIFFUSE.png (RGB albedo map)
│   │   # 	METALLIC.png (metallic map)
│   │   # 	NORMAL.png (tangent normal map)
│   │   # 	ROUGHNESS.png (roughness map)
|   |   # Also contains diffuse texture maps that will be used when some object state changes happen, e.g. cooked, burnt, or soaked.
│   │   DIFFUSE.encrypted.png
│   │   METALLIC.encrypted.png	
│   │   NORMAL.encrypted.png
│   │   ROUGHNESS.encrypted.png
│   │   DIFFUSE_Frozen.encrypted.png
│   │   DIFFUSE_Cooked.encrypted.png
│   │   DIFFUSE_Burnt.encrypted.png
│   │   DIFFUSE_Soaked.encrypted.png
│   │   DIFFUSE_ToggledOn.encrypted.png
│
└───misc
│   │   # contains bounding box information of the object, its stable orientations, object state-related link annotation (e.g. toggle button, water/heat/cold source, slicer/cleaning_tool, etc)
│   │   metadata.json
│   │   # contains the object’s material annotation of what kinds of material each link can have.
│   │   material_groups.json 
│   │   # contains info about the minimum volume oriented bounding box that best fits the object
│   │   mvbb_meta.json 
│
└───visualizations
│   │   # An image and a video of the object rendered with iG renderer
│   │   00.png
│   │   OBJECT_NAME.mp4

```



### Adding other objects to iGibson
We provide detailed instructions and scripts to import your own objects (non-articulated) into iGibson. 

Instruction can be found here: [External Objects](https://github.com/StanfordVL/iGibson/blob/master/igibson/utils/data_utils/ext_object). 


### Examples
In this example, we import a few objects into iGibson. The code can be found here: [igibson/examples/objects/load_objects.py](https://github.com/StanfordVL/iGibson/blob/master/igibson/examples/objects/load_objects.py).

```{literalinclude} ../igibson/examples/objects/load_objects.py
:language: python
```