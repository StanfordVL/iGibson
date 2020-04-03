Dataset
==========================================

TODO: @fei
Make sure everything is up-to-date.

Full Gibson Environment Dataset consists of 572 models and 1440 floors. We cover a diverse set of models including households, offices, hotels, venues, museums, hospitals, construction sites, etc. A diverse set of visualization of all spaces in Gibson can be seen [here](http://gibsonenv.stanford.edu/database/).
 
<img src=../../misc/spaces.png width="800">


Download Gibson Database of Spaces
----

The link will first take you to the license agreement and then to the data.

[[ Download the full Gibson Database of Spaces ]](https://goo.gl/forms/OxAQHbl1v97BJ3Sg1)  [[ checksums ]](https://github.com/StanfordVL/GibsonEnv/wiki/Checksum-Values-for-Data.md)

License Note: The dataset license is included in the above link. The license in this repository covers only the provided software.

**Stanford 2D-3D-Semantics Dataset:** the download link of 2D-3D-Semantics as Gibson asset files is included in the [same link ](https://goo.gl/forms/OxAQHbl1v97BJ3Sg1) as above. 

**Matterport3D Dataset:** Please fill and sign the corresponding [Terms of Use agreement](http://dovahkiin.stanford.edu/matterport/public/MP_TOS.pdf) form and send it to [matterport3d@googlegroups.com](matterport3d@googlegroups.com). Please put "use with GIBSON simulator" in your email. You'll then recieve a python script via email in response. Use the invocation `python download_mp.py --task_data gibson -o .` with the received script to download the data (39.09GB). Matterport3D webpage: [link](https://niessner.github.io/Matterport/).


Dataset Metadata
----

Each space in the database has some metadata with the following attributes associated with it. The metadata is available in this [JSON file](https://raw.githubusercontent.com/StanfordVL/GibsonEnv/master/gibson/data/data.json). 
```
id                      # the name of the space, e.g. ""Albertville""
area                    # total metric area of the building, e.g. "266.125" sq. meters
floor                   # number of floors in the space, e.g. "4"
navigation_complexity   # navigation complexity metric, e.g. "3.737" (see the paper for definition)
room                    # number of rooms, e.g. "16"
ssa                     # Specific Surface Area (A measure of clutter), e.g. "1.297" (see the paper for definition)
split_full              # if the space is in train/val/test/none split of Full partition 
split_full+             # if the space is in train/val/test/none split of Full+ partition 
split_medium            # if the space is in train/val/test/none split of Medium partition 
split_tiny              # if the space is in train/val/test/none split of Tiny partition 
```

Dataset Format
----

Each space in the database has its own folder. All the modalities and metadata for each space are contained in that folder. 
```
/pano
  /points                 # camera metadata
  /rgb                    # rgb images
  /mist                   # depth images
mesh.obj                  # 3d mesh
mesh_z_up.obj             # 3d mesh for physics engine
camera_poses.csv          # camera locations
semantic.obj (optional)   # 3d mesh with semantic annotation
```


Dataset Metrics
-------------



**Floor Number** Total number of floors in each model.

We calculate floor numbers using distinctive camera locations. We use `sklearn.cluster.DBSCAN` to cluster these locations by height and set minimum cluster size to `5`. This means areas with at least `5` sweeps are treated as one single floor. This helps us capture small building spaces such as backyard, attics, basements.

**Area** Total floor area of each model.

We calculate total floor area by summing up area of each floor. This is done by sampling point cloud locations based on floor height, and fitting a `scipy.spatial.ConvexHull` on sample locations.

**SSA** Specific surface area. 

The ratio of inner mesh surface and volume of convex hull of the mesh. This is a measure of clutter in the models: if the inner space is placed with large number of furnitures, objects, etc, the model will have high SSA. 

**Navigation Complexity** The highest complexity of navigating between arbitrary points within the model.

We sample arbitrary point pairs inside the model, and calculate `A∗` navigation distance between them. `Navigation Complexity` is equal to `A*` distance divide by `straight line distance` between the two points. We compute the highest navigation complexity for every model. Note that all point pairs are sample within the *same floor*.

**Subjective Attributes**

We examine each model manually, and note the subjective attributes of them. This includes their furnishing style, house shapes, whether they have long stairs, etc.

