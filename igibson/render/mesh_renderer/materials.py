import json
import math
import os
import random
import uuid

import numpy as np

import igibson
from igibson.object_states.factory import TEXTURE_CHANGE_PRIORITY


class Material(object):
    """
    Material class used for iG renderer
    """

    def __init__(
        self,
        material_type="color",
        kd=[0.5, 0.5, 0.5],
        texture_id=None,
        metallic_texture_id=None,
        roughness_texture_id=None,
        normal_texture_id=None,
        transform_param=[1, 1, 0],
    ):
        """
        :param material_type: color or texture
        :param kd: color parameters
        :param texture_id: albedo texture id
        :param metallic_texture_id: metallic texture id
        :param roughness_texture_id: roughness texture id
        :param normal_texture_id: normal texture id
        :param transform_param: x scale, y scale, rotation
        """
        self.material_type = material_type
        self.kd = kd
        self.texture_id = texture_id
        self.metallic_texture_id = metallic_texture_id
        self.roughness_texture_id = roughness_texture_id
        self.normal_texture_id = normal_texture_id
        self.transform_param = transform_param

    def is_texture(self):
        """
        Returns whether this material has texture (as opposed to single color)
        """
        return self.material_type == "texture"

    def is_pbr_texture(self):
        """
        Returns whether this material can be used for PBR
        """
        return (
            self.material_type == "texture"
            and self.metallic_texture_id is not None
            and self.roughness_texture_id is not None
            and self.normal_texture_id is not None
        )

    def __str__(self):
        return (
            "Material(material_type: {}, texture_id: {}, metallic_texture_id:{}, roughness_texture_id:{}, "
            "normal_texture_id:{}, color: {})".format(
                self.material_type,
                self.texture_id,
                self.metallic_texture_id,
                self.roughness_texture_id,
                self.normal_texture_id,
                self.kd,
            )
        )

    def __repr__(self):
        return self.__str__()


class ProceduralMaterial(Material):
    def __init__(
        self,
        material_folder="",
        material_type="texture",
        kd=[0.5, 0.5, 0.5],
        texture_id=None,
        metallic_texture_id=None,
        roughness_texture_id=None,
        normal_texture_id=None,
    ):
        """
        :param material_type: color or texture
        :param kd: color parameters
        :param texture_id: albedo texture id
        :param metallic_texture_id: metallic texture id
        :param roughness_texture_id: roughness texture id
        :param normal_texture_id: normal texture id
        :param transform_param: x scale, y scale, rotation
        """
        super(ProceduralMaterial, self).__init__(
            material_type=material_type,
            kd=kd,
            texture_id=texture_id,
            metallic_texture_id=metallic_texture_id,
            roughness_texture_id=roughness_texture_id,
            normal_texture_id=normal_texture_id,
        )
        self.material_folder = material_folder
        # a list of object states that can change the material
        self.states = []
        # mapping from object states to transformed material filenames
        # that should be applied when the states are evaluated to be True
        self.texture_filenames = {}
        # mapping from object states to loaded texture ids from the material
        # filenames stored in self.texture_filenames. Will be popuated by
        # load_procedural_material in mesh_renderer_cpu.py
        self.texture_ids = {}
        # default texture id. Will be populated by load_procedural_material
        # in mesh_renderer_cpu.py
        self.default_texture_id = None

        # after material changes, request_update is set to True so that
        # optimized renderer can update the texture ids
        self.request_update = False
        # keep tracks of all the requests since the last texture update
        self.requests = []

    def __str__(self):
        return (
            "ProceduralMaterial(material_type: {}, texture_id: {}, "
            "metallic_texture_id: {}, roughness_texture_id: {}, "
            "normal_texture_id: {}, color: {})".format(
                self.material_type,
                self.texture_id,
                self.metallic_texture_id,
                self.roughness_texture_id,
                self.normal_texture_id,
                self.kd,
            )
        )

    def request_texture_change(self, state):
        self.requests.append(state)

    def add_state(self, state):
        self.states.append(state)

    def lookup_or_create_transformed_texture(self):
        has_encrypted_texture = os.path.exists(os.path.join(self.material_folder, "DIFFUSE.encrypted.png"))

        suffix = ".encrypted.png" if has_encrypted_texture else ".png"
        diffuse_tex_filename = os.path.join(self.material_folder, "DIFFUSE{}".format(suffix))

        for state in self.states:
            diffuse_tex_filename_lookup = os.path.join(
                self.material_folder, "DIFFUSE_{}{}".format(state.__name__, suffix)
            )
            if os.path.exists(diffuse_tex_filename_lookup):
                self.texture_filenames[state] = diffuse_tex_filename_lookup
            else:
                if has_encrypted_texture:
                    raise ValueError("cannot create transformed textures for encrypted texture")
                save_path = os.path.join(igibson.ig_dataset_path, "tmp")
                if not os.path.isdir(save_path):
                    os.makedirs(save_path, exist_ok=True)
                diffuse_tex_filename_transformed = os.path.join(save_path, str(uuid.uuid4()) + ".png")
                state.create_transformed_texture(
                    diffuse_tex_filename=diffuse_tex_filename,
                    diffuse_tex_filename_transformed=diffuse_tex_filename_transformed,
                )
                self.texture_filenames[state] = diffuse_tex_filename_transformed

    def update(self):
        if len(self.requests) == 0:
            texture_id = self.default_texture_id
        else:
            requests = sorted(self.requests, key=lambda state: TEXTURE_CHANGE_PRIORITY[state])
            highest_priority_state = requests[-1]
            texture_id = self.texture_ids[highest_priority_state]

        # Request update if the texture_id doesn't match the current one
        if self.texture_id != texture_id:
            self.texture_id = texture_id
            self.request_update = True

        self.requests.clear()


class RandomizedMaterial(Material):
    """
    Randomized Material class for material randomization
    """

    def __init__(
        self,
        material_classes,
        material_type="texture",
        kd=[0.5, 0.5, 0.5],
        texture_id=None,
        metallic_texture_id=None,
        roughness_texture_id=None,
        normal_texture_id=None,
    ):
        """
        :param material_classes: a list of material classes
        :param material_type: color or texture
        :param kd: color parameters
        :param texture_id: albedo texture id
        :param metallic_texture_id: metallic texture id
        :param roughness_texture_id: roughness texture id
        :param normal_texture_id: normal texture id
        :param transform_param: x scale, y scale, rotation
        """
        super(RandomizedMaterial, self).__init__(
            material_type=material_type,
            kd=kd,
            texture_id=texture_id,
            metallic_texture_id=metallic_texture_id,
            roughness_texture_id=roughness_texture_id,
            normal_texture_id=normal_texture_id,
        )
        # a list of material classes, str
        self.material_classes = self.postprocess_material_classes(material_classes)
        # a dict that maps from material class to a list of material files
        # {
        #     'wood': [
        #         {
        #             'diffuse': diffuse_path,
        #             'metallic': metallic_path,
        #             'roughness': None
        #             'normal': normal_path
        #         },
        #         {
        #             ...
        #         }
        #     ],
        #     'metal': [
        #         ...
        #     ]
        # }
        self.material_files = self.get_material_files()
        # a dict that maps from material class to a list of texture ids
        # {
        #     'wood': [
        #         {
        #             'diffuse': 25,
        #             'metallic': 26,
        #             'roughness': None
        #             'normal': 27
        #         },
        #         {
        #             ...
        #         }
        #     ],
        #     'metal': [
        #         ...
        #     ]
        # }

        # WILL be populated when the texture is actually loaded
        self.material_ids = None

        self.random_class = None
        self.random_instance = None

    def postprocess_material_classes(self, material_classes):
        """
        Postprocess material classes.
        We currently do not have all the annotated materials, so we will need
        to convert the materials that we don't have to their closest neighbors
        that we do have.

        :param material_classes: original material classes
        :return material_classes: postprocessed material classes
        """
        for i in range(len(material_classes)):
            material_class = material_classes[i]
            if material_class in ["rock"]:
                material_class = "rocks"
            elif material_class in ["fence", ""]:
                material_class = "wood"
            elif material_class in ["flower", "leaf"]:
                material_class = "moss"
            elif material_class in ["cork"]:
                material_class = "chipboard"
            elif material_class in ["mirror", "glass", "screen"]:
                material_class = "metal"
            elif material_class in ["painting", "picture"]:
                material_class = "paper"
            material_classes[i] = material_class
        return material_classes

    def get_material_files(self):
        """
        Retrieve the material files from the material dataset,
        given the material classes

        :return material_files: a dict that maps material_class to material files
        """
        material_dir = os.path.join(igibson.ig_dataset_path, "materials")
        material_json_file = os.path.join(material_dir, "materials.json")
        assert os.path.isfile(material_json_file), "cannot find material files: {}".format(material_json_file)
        with open(material_json_file) as f:
            all_materials = json.load(f)

        material_files = {}
        for material_class in self.material_classes:
            material_files[material_class] = []
            assert material_class in all_materials, "unknown material class: {}".format(material_class)

            # append igibson.ig_dataset_path/materials to the beginning
            for material_instance in all_materials[material_class]:
                for key in all_materials[material_class][material_instance]:
                    value = all_materials[material_class][material_instance][key]
                    if value is not None:
                        value = os.path.join(material_dir, value)
                    all_materials[material_class][material_instance][key] = value
            material_files[material_class] = list(all_materials[material_class].values())
        return material_files

    def randomize(self):
        """
        Randomize the material by randomly sampling a material id that belongs
        to one of the material classes. All potential materials have already
        been loaded to memory.
        """
        if self.material_ids is None:
            return
        self.random_class = random.choice(list(self.material_ids.keys()))
        self.random_instance = random.choice(self.material_ids[self.random_class])
        self.texture_id = self.random_instance["diffuse"]
        self.metallic_texture_id = self.random_instance["metallic"]
        self.roughness_texture_id = self.random_instance["roughness"]
        self.normal_texture_id = self.random_instance["normal"]

        # scaling by 4 is typically good
        scale = np.random.normal(loc=4, scale=1)
        # scaling should be at least 2.
        scale = max(scale, 2)
        rotation = random.randint(0, 3) * math.pi / 2.0
        self.transform_param = [scale, scale, rotation]

    def __str__(self):
        return (
            "RandomizedMaterial(material_type: {}, texture_id: {}, "
            "metallic_texture_id: {}, roughness_texture_id: {}, "
            "normal_texture_id: {}, color: {}, material_classes: {})".format(
                self.material_type,
                self.texture_id,
                self.metallic_texture_id,
                self.roughness_texture_id,
                self.normal_texture_id,
                self.kd,
                self.material_classes,
            )
        )
