import os
import gibson2
from gibson2.objects.object_base import Object
import pybullet as p
import json
import numpy as np
import gibson2.utils.transform_utils as T


class YCBObject(Object):
    """
    YCB Object from assets/models/ycb
    Reference: https://www.ycbbenchmarks.com/
    """

    def __init__(self, name, scale=1.0, mass=0.1):
        super(YCBObject, self).__init__()
        self.visual_filename = os.path.join(gibson2.assets_path, 'models', 'ycb', name,
                                            'textured_simple.obj')
        self.collision_filename = os.path.join(gibson2.assets_path, 'models', 'ycb', name,
                                               'textured_simple_vhacd.obj')
        self.scale = scale
        self.mass = mass
        self.scale = [scale] * 3 if isinstance(scale, float) else scale

        # Load metadata info
        metadata_fpath = os.path.join(gibson2.assets_path, 'models', 'ycb', name, 'metadata.json')
        with open(metadata_fpath, 'r') as f:
            self.metadata = json.load(f)

        # Store relevant info
        self.radius = self.metadata["radius"] * scale
        self.height = self.metadata["height"] * scale
        self.bottom_offset = self.metadata["bottom_offset"] * scale

        # We want to load the metadata for obj sampling
        self.sampling_surfaces = {}
        surfaces = self.metadata.get('valid_surfaces', None)
        if surfaces is not None:
            for surface in surfaces:
                self.sampling_surfaces[surface["name"]] = {
                    "link": surface["link"],
                    "offset_from_base": np.array(surface["offset_from_base"]) * scale,
                    "size": np.array(surface["size"]) * scale,
                    "max_height": np.array(surface["max_height"]) * scale,
                    "prob": surface["prob"],
                }

    def _load(self):
        collision_id = p.createCollisionShape(p.GEOM_MESH,
                                              fileName=self.collision_filename,
                                              meshScale=self.scale)
        visual_id = p.createVisualShape(p.GEOM_MESH,
                                        fileName=self.visual_filename,
                                        meshScale=self.scale)

        body_id = p.createMultiBody(baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=visual_id,
                                    basePosition=[0.2, 0.2, 1.5],
                                    baseMass=self.mass)
        return body_id

    def sample_obj_position(self, obj_radius=0.0, obj_height=0.0, bottom_offset=0.0, surfaces=None):
        """
        Samples a valid location on / in this object for another object defined by @obj_radius and @obj_height to
        be placed.

        Args:
            obj_radius (float): Radius of the object to be placed somewhere on / in this object. Defaults to 0, which
                corresponds to no sampling compensation
            obj_height (float): Height of the object to be placed somewhere on / in this object. Defaults to 0, which
                corresponds to all sampling surfaces being considered valid
            bottom_offset (float): Distance from center of object to bottom surface of object being placed. Defaults to
                0, which corresponds to center of object being assumed to be the bottom surface (presumably this should
                be negative)
            surfaces (None or str or list of str): If specified, will sample location specifically from this surface(s).
                Otherwise, will be randomly sampled.

        Returns:
            np.array: (x,y,z) global coordinates sampled representing a valid location on / in this object
        """
        # Create valid surfaces
        if type(surfaces) is str:
            surfaces = [surfaces]
        elif surfaces is None:
            surfaces = list(self.sampling_surfaces.keys())
        else:
            # Assumed to be list or tuple
            surfaces = list(surfaces)

        # Get probabilities for possible surfaces and normalize
        probs = np.array([self.sampling_surfaces[s]["prob"] for s in surfaces])
        probs /= np.sum(probs)

        # Sample surface location
        surface_names = np.random.choice(
            a=surfaces,
            size=len(surfaces),
            replace=False,
            p=probs,
        )

        # Next sample appropriate location
        success = False
        sampled_location = np.zeros(3)
        for surface_name in surface_names:
            # See if this surface is tall enough to accommodate the object
            surface = self.sampling_surfaces[surface_name]
            if obj_height > surface["max_height"]:
                # Object is too tall, try the next surface
                print(f"obj too tall for sampled surface {surface_name}")
                continue
            elif obj_radius * 2 > surface["size"][0] or obj_radius * 2 > surface["size"][1]:
                # Object is too wide, try next surface
                print(f"obj too wide for sampled surface {surface_name}; obj radius: {obj_radius}, surface_size: {surface['size']}")
                continue
            else:
                # Sample location
                for i in range(2):      # x, y
                    sampled_location[i] = surface["offset_from_base"][i] + \
                                          (surface["size"][i] - 2 * obj_radius) * (np.random.rand() - 0.5)
                # Set height
                sampled_location[2] = surface["offset_from_base"][2] - bottom_offset
                # Rotate the x, y values according to init_ori (yaw value, rotation about z axis)
                pos, quat = np.array(self.get_position()), np.array(self.get_orientation())
                ori = T.mat2euler(T.quat2mat(quat))
                z_rot = ori[2]
                sampled_location[:2] = np.array([[np.cos(z_rot), -np.sin(z_rot)],[np.sin(z_rot), np.cos(z_rot)]]) @ sampled_location[:2]
                # Offset this value by the base global position
                sampled_location += pos
                success = True
                # Stop looping, we've found a valid sampling location
                break

        # If no success, we couldn't find a valid sample ): Raise an error to let user know
        assert success, "No valid sampling location could be found!"

        # Return the sampled position
        return sampled_location
