import datetime
import itertools
import json
import os
import random
from typing import Callable, Dict, Sequence, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from bddl.object_taxonomy import ObjectTaxonomy
from PIL import Image
from tqdm import tqdm

import igibson
from igibson import object_states
from igibson.envs.igibson_env import iGibsonEnv
from igibson.object_states.object_state_base import AbsoluteObjectState
from igibson.objects.stateful_object import StatefulObject
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.utils import semantics_utils
from igibson.utils.constants import MAX_INSTANCE_COUNT, SemanticClass

OBJECT_TAXONOMY = ObjectTaxonomy(hierarchy_type="all")


class DerivativeDatasetPipeline:
    def __init__(
        self,
        scene_id: str,
        prefix: str,
        output_path: str,
        render_width: int,
        render_height: int,
        debug_filters: bool,
        debug_filter_images: bool,
        export_rgb: bool,
        export_depth: bool,
        export_seg: bool,
        export_metadata: bool,
        states_to_include_in_metadata: Sequence[str],
        use_wordnet_category: bool,
        requested_images: int,
        images_per_perturbation: int,
        max_attempts_per_perturbation: int,
        max_depth: int,  # meters
        perturbers: Dict[str, Dict[str, Union[float, Callable]]],
        generators: Dict[str, Dict[str, Union[float, Callable]]],
        filters: Dict[str, Callable],
    ):
        self.scene_id = scene_id
        self.prefix = prefix
        self.output_path = output_path

        self.render_width = render_width
        self.render_height = render_height

        self.debug_filters = debug_filters
        self.debug_filter_images = debug_filter_images

        self.export_rgb = export_rgb
        self.export_depth = export_depth
        self.export_seg = export_seg
        self.export_metadata = export_metadata
        self.states_to_include_in_metadata = states_to_include_in_metadata
        self.use_wordnet_category = use_wordnet_category

        self.requested_images = requested_images
        self.images_per_perturbation = images_per_perturbation
        self.max_attempts_per_perturbation = max_attempts_per_perturbation

        self.max_depth = max_depth

        self.perturbers = perturbers
        self.generators = generators
        self.filters = filters

        self.filter_img_idx = {f: 0 for f in self.filters.keys()}
        self.total_image_count = 0
        self.total_annotation_count = 0

        self.taxonomy = nx.relabel.convert_node_labels_to_integers(
            OBJECT_TAXONOMY.taxonomy, ordering="sorted", label_attribute="original_label"
        )

        hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
        hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
        light_modulation_map_filename = os.path.join(
            igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
        )
        background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")

        rendering_settings = MeshRendererSettings(
            optimized=False,
            fullscreen=False,
            env_texture_filename=hdr_texture,
            env_texture_filename2=hdr_texture2,
            env_texture_filename3=background_texture,
            light_modulation_map_filename=light_modulation_map_filename,
            enable_shadow=True,
            enable_pbr=True,
            msaa=False,
            light_dimming_factor=1.0,
        )
        self.env = iGibsonEnv(
            scene_id=self.scene_id,
            mode="headless",
            config_file={
                "image_width": self.render_width,
                "image_height": self.render_height,
                "vertical_fov": 60,
                "scene": "igibson",
            },
            rendering_settings=rendering_settings,
        )

        for _ in range(100):
            self.env.step(None)

        # Create metadata file
        self.metadata_path = os.path.join(self.output_path, f"{prefix}labels.json")
        info = {
            "year": "2022",
            "version": "1.0",
            "description": "Exported from OmniGibson",
            "contributor": "cgokmen",
            "url": "https://behavior.stanford.edu",
            "date_created": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.metadata_path, "w") as f:
            json.dump({"info": info, "images": [], "annotations": [], "categories": []}, f)

    def run_filters(self, objs_of_interest):
        for filter_name, filter_fn in self.filters.items():
            if not filter_fn(self.env, objs_of_interest):
                self.filter_img_idx[filter_name] += 1

                if self.debug_filters and random.uniform(0, 1) < 0.01:
                    x = np.arange(len(self.filter_img_idx))
                    h = list(self.filter_img_idx.values())
                    label = list(self.filter_img_idx.keys())
                    plt.bar(x, h)
                    plt.xticks(x, label, rotation=45)
                    plt.savefig(os.path.join(self.output_path, "filters.png"))

                    if self.debug_filter_images:
                        filter_img_path = os.path.join(self.output_path, "filters", filter_name)
                        os.makedirs(filter_img_path, exist_ok=True)
                        (rgb,) = self.env.simulator.renderer.render(("rgb",))
                        rgb_img = Image.fromarray(np.uint8(rgb[:, :, :3] * 255))
                        rgb_img.save(os.path.join(filter_img_path, f"{self.filter_img_idx[filter_name]}.png"))

                return False

        return True

    def save_images(self, objs_of_interest):
        img_id = self.total_image_count
        rgb, segmask, threed = self.env.simulator.renderer.render(("rgb", "ins_seg", "3d"))

        rgb_arr = np.uint8(rgb[:, :, :3] * 255)
        rgb_img = Image.fromarray(rgb_arr)
        depth = np.clip(-threed[:, :, 2:3], 0, self.max_depth) / self.max_depth
        depth_arr = np.uint8(depth[..., 0] * 255)
        depth_img = Image.fromarray(depth_arr)

        seg = np.round(segmask[:, :, 0] * MAX_INSTANCE_COUNT).astype(int)
        body_ids = self.env.simulator.renderer.get_pb_ids_for_instance_ids(seg)
        unique_body_ids, lowered_body_ids = np.unique(body_ids, return_inverse=True)
        seg_arr = np.uint8(lowered_body_ids.reshape(body_ids.shape))
        seg_img = Image.fromarray(seg_arr)

        out_dir = self.output_path
        filename = f"{self.prefix}{img_id}.png"

        if self.export_rgb:
            rgb_dir = os.path.join(out_dir, "rgb")
            os.makedirs(rgb_dir, exist_ok=True)
            rgb_img.save(os.path.join(rgb_dir, filename))

        if self.export_depth:
            depth_dir = os.path.join(out_dir, "depth")
            os.makedirs(depth_dir, exist_ok=True)
            depth_img.save(os.path.join(depth_dir, filename))

        if self.export_seg:
            seg_dir = os.path.join(out_dir, "seg")
            os.makedirs(seg_dir, exist_ok=True)
            seg_img.save(os.path.join(seg_dir, filename))

        if self.export_metadata:
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)

            cam_pos, cam_orn = self.env.simulator.renderer.get_camera_pose()
            metadata["images"].append(
                {
                    "id": img_id,
                    "width": self.render_width,
                    "height": self.render_height,
                    "file_name": f"{self.prefix}{img_id}.png",
                    "scene_id": self.scene_id,
                    "camera_position": cam_pos.tolist(),
                    "camera_orientation": cam_orn.tolist(),
                    "date_created": datetime.datetime.now().isoformat(timespec="seconds"),
                }
            )

        obj_body_ids = [x for obj in objs_of_interest for x in obj.get_body_ids()]
        found_obj_body_ids = set(unique_body_ids) & set(obj_body_ids)

        # crop_out_dir = os.path.join(self.output_path, "cropped")
        for crop_id, body_id in enumerate(found_obj_body_ids):
            obj = self.env.simulator.scene.objects_by_id[body_id]

            # Get the pixels belonging to this object.
            this_obj_pixels = body_ids == body_id

            # Get the crop bounding box positions.
            rows = np.any(this_obj_pixels, axis=1)
            cols = np.any(this_obj_pixels, axis=0)
            bb_rmin, bb_rmax = np.where(rows)[0][[0, -1]]
            bb_cmin, bb_cmax = np.where(cols)[0][[0, -1]]
            bb_h = bb_rmax - bb_rmin
            bb_w = bb_cmax - bb_cmin

            if self.export_metadata:
                # Get the semantic ID first
                if self.use_wordnet_category:
                    class_name_with_dot = OBJECT_TAXONOMY.get_class_name_from_igibson_category(obj.category)
                    if class_name_with_dot is None:
                        class_name_with_dot = "entity.n.01"
                    (class_id,) = [
                        node
                        for node, label in self.taxonomy.nodes(data="original_label")
                        if label == class_name_with_dot
                    ]
                    class_name = class_name_with_dot.split(".")[0]
                else:
                    class_name = obj.category
                    if class_name in semantics_utils.CLASS_NAME_TO_CLASS_ID:
                        class_id = semantics_utils.CLASS_NAME_TO_CLASS_ID[class_name]
                    else:
                        class_id = SemanticClass.SCENE_OBJS
                        class_name = SemanticClass.SCENE_OBJS.name

                states = {}
                if isinstance(obj, StatefulObject):
                    for state_type in self.states_to_include_in_metadata:
                        states[state_type.__name__] = obj.states[state_type].get_value()

                # Make sure that the semantic ID is listed
                if class_id not in [x["id"] for x in metadata["categories"]]:
                    metadata["categories"].append(
                        {
                            "id": class_id,
                            "name": class_name,
                        }
                    )

                # Add the annotation
                metadata["annotations"].append(
                    {
                        "id": self.total_annotation_count,
                        "image_id": img_id,
                        "category_id": class_id,
                        "bbox": (int(bb_cmin), int(bb_rmin), int(bb_w), int(bb_h)),
                        "iscrowd": 0,
                        "area": int(np.count_nonzero(this_obj_pixels)),
                        "segmentation": [],  # [tuple(x) for x in np.argwhere(this_obj_pixels)]
                        "states": states,
                    }
                )

                self.total_annotation_count += 1

            # This code is commented out because it has been replaced by the COCO-based
            # cropping dataset implementation in PyTorch.
            #
            # # Add the margins
            # rmin = np.clip(bb_rmin - self.crop_margin, 0, self.render_height - 1)
            # rmax = np.clip(bb_rmax + self.crop_margin, 0, self.render_height - 1)
            # cmin = np.clip(bb_cmin - self.crop_margin, 0, self.render_width - 1)
            # cmax = np.clip(bb_cmax + self.crop_margin, 0, self.render_width - 1)
            #
            # # Crop the images at the bounding box borders.
            # cropped_rgb = Image.fromarray(rgb_arr[rmin : rmax + 1, cmin : cmax + 1])
            # cropped_depth = Image.fromarray(depth_arr[rmin : rmax + 1, cmin : cmax + 1])
            # cropped_seg = Image.fromarray(seg_arr[rmin : rmax + 1, cmin : cmax + 1])
            #
            # # Prepare labelled directories.
            # label = "open" if obj.states[object_states.Open].get_value() else "closed"
            #
            # if self.export_cropped_rgb:
            #     labeled_rgb_dir = os.path.join(crop_out_dir, "rgb", label)
            #     os.makedirs(labeled_rgb_dir, exist_ok=True)
            #     cropped_rgb.save(os.path.join(labeled_rgb_dir, f"{self.prefix}{img_id}_{crop_id}.png"))
            #
            # if self.export_cropped_depth:
            #     labeled_depth_dir = os.path.join(crop_out_dir, "depth", label)
            #     os.makedirs(labeled_depth_dir, exist_ok=True)
            #     cropped_depth.save(os.path.join(labeled_depth_dir, f"{self.prefix}{img_id}_{crop_id}.png"))
            #
            # if self.export_cropped_seg:
            #     labeled_seg_dir = os.path.join(crop_out_dir, "seg", label)
            #     os.makedirs(labeled_seg_dir, exist_ok=True)
            #     cropped_seg.save(os.path.join(labeled_seg_dir, f"{self.prefix}{img_id}_{crop_id}.png"))

        if self.export_metadata:
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f)

        self.total_image_count += 1

    def generate(self):
        """
        Prompts the user to select any available interactive scene and loads it.
        Shows how to load directly scenes without the Environment interface
        Shows how to sample points in the scene by room type and how to compute geodesic distance and the shortest path
        """
        perturbers, perturber_probs = zip(
            *[
                (perturber_info["perturber"], perturber_info["probability"])
                for perturber_info in self.perturbers.values()
            ]
        )
        perturbers = list(perturbers)
        perturber_probs = list(perturber_probs)
        assert np.isclose(np.sum(perturber_probs), 1), "Perturber probabilities should add up to 1."

        with tqdm(total=self.requested_images) as pbar:
            while self.total_image_count < self.requested_images:
                self.env.simulator.scene.reset_scene_objects()
                for _ in range(100):
                    self.env.step(None)

                (perturber,) = random.choices(perturbers, weights=perturber_probs, k=1)
                if perturber is not None:
                    objs_of_interest = perturber(self.env)
                else:
                    objs_of_interest = self.env.scene.get_objects()
                self.env.simulator.sync(force_sync=True)

                perturbation_image_count = 0
                attempts = 0

                generators, generator_probs = zip(
                    *[
                        (generator_info["generator"], generator_info["probability"])
                        for generator_info in self.generators.values()
                    ]
                )
                generators = list(generators)
                generator_probs = list(generator_probs)
                assert np.isclose(np.sum(generator_probs), 1), "Generator probabilities should add up to 1."

                while (
                    perturbation_image_count < self.images_per_perturbation
                    and attempts < self.max_attempts_per_perturbation
                ):
                    pbar.set_postfix({"attempts": attempts})
                    attempts += 1
                    (generator,) = random.choices(generators, weights=generator_probs, k=1)

                    camera_pos, camera_target, camera_up = generator(self.env, objs_of_interest)
                    self.env.simulator.renderer.set_camera(camera_pos, camera_target, camera_up)

                    # v = VisualMarker(radius=0.1)
                    # env.simulator.import_object(v)
                    # v.set_position(camera_pos)

                    if not self.run_filters(objs_of_interest):
                        continue

                    self.save_images(objs_of_interest)

                    perturbation_image_count += 1
                    pbar.update()

        print(self.filter_img_idx)

        self.env.simulator.disconnect()
