import collections
import itertools
import logging
import os
import random
from sys import platform

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from igibson import object_states
from igibson.envs.igibson_env import iGibsonEnv
from igibson.objects.visual_marker import VisualMarker
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.utils.derivative_dataset import filters, generators, perturbers

RENDER_WIDTH = 1080
RENDER_HEIGHT = 720

REQUESTED_IMAGES = 100
IMAGES_PER_PERTURBATION = 10
MAX_ATTEMPTS_PER_PERTURBATION = 1000

DEBUG_FILTERS = True
DEBUG_FILTER_IMAGES = True

OUTPUT_DIR = r"C:\Users\cgokmen\research\derivative_dataset_tests"

GENERATORS = [
    # generators.uniform_generator,
    generators.object_targeted_generator,
]

PERTURBERS = [
    perturbers.object_boolean_state_randomizer(object_states.Open),
]

FILTERS = {
    "no_collision": filters.point_in_object_filter(),
    "no_openable_objects_fov": filters.no_relevant_object_in_fov_filter(object_states.Open, min_bbox_vertices_in_fov=4),
    "no_openable_objects_img": filters.no_relevant_object_in_img_filter(object_states.Open, threshold=0.05),
    "some_objects_closer_than_10cm": filters.too_close_filter(min_dist=0.1, max_allowed_fraction_outside_threshold=0.1),
    # At least 70% of the image between 30cm and 2m away
    # "too_many_too_close_far_objects": filters.too_close_filter(min_dist=0.5, max_dist=3., max_allowed_fraction_outside_threshold=0.3),
    # No more than 50% of the image should consist of wall/floor/ceiling
    "too_much_structure": filters.too_much_structure(max_allowed_fraction_of_structure=0.5),
    # More than 33% of the image should not be the same object.
    "too_much_of_the_same_object": filters.too_much_of_same_object_in_fov_filter(threshold=0.5),
}

FILTER_IMG_IDX = {f: 0 for f in FILTERS}


def run_filters(env, objs_of_interest):
    for filter_name, filter_fn in FILTERS.items():
        if not filter_fn(env, objs_of_interest):
            print("Failed ", filter_name)
            FILTER_IMG_IDX[filter_name] += 1

            if DEBUG_FILTERS and random.uniform(0, 1) < 0.01:
                x = np.arange(len(FILTER_IMG_IDX))
                h = list(FILTER_IMG_IDX.values())
                l = list(FILTER_IMG_IDX.keys())
                plt.bar(x, h)
                plt.xticks(x, l)
                plt.show()

                if DEBUG_FILTER_IMAGES:
                    filter_img_path = os.path.join(OUTPUT_DIR, "filters", filter_name)
                    os.makedirs(filter_img_path, exist_ok=True)
                    (rgb,) = env.simulator.renderer.render(("rgb"))
                    rgb_img = Image.fromarray(np.uint8(rgb[:, :, :3] * 255))
                    rgb_img.save(os.path.join(filter_img_path, f"{FILTER_IMG_IDX[filter_name]}.png"))

            return False

    return True


def main(headless=False, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads it.
    Shows how to load directly scenes without the Environment interface
    Shows how to sample points in the scene by room type and how to compute geodesic distance and the shortest path
    """
    scene_id = "Rs_int"
    settings = MeshRendererSettings(enable_shadow=True, msaa=False, optimized=False)
    if platform == "darwin":
        settings.texture_scale = 0.5
    env = iGibsonEnv(
        scene_id=scene_id,
        mode="gui_interactive",
        config_file={
            "image_width": RENDER_WIDTH,
            "image_height": RENDER_HEIGHT,
            "vertical_fov": 60,
            "scene": "igibson",
        },
        rendering_settings=settings,
    )

    for _ in range(100):
        env.step(None)

    total_image_count = 0
    perturbers = itertools.cycle(PERTURBERS)
    while total_image_count < REQUESTED_IMAGES:
        perturber = next(perturbers)
        env.simulator.scene.reset_scene_objects()
        for _ in range(100):
            env.step(None)

        objs_of_interest = perturber(env)
        env.simulator.sync(force_sync=True)

        perturbation_image_count = 0
        attempts = 0
        generators = itertools.cycle(GENERATORS)
        while perturbation_image_count < IMAGES_PER_PERTURBATION and attempts < MAX_ATTEMPTS_PER_PERTURBATION:
            print("Attempt ", attempts)
            attempts += 1
            generator = next(generators)

            camera_pos, camera_target, camera_up = generator(env, objs_of_interest)
            env.simulator.renderer.set_camera(camera_pos, camera_target, camera_up)

            # v = VisualMarker(radius=0.1)
            # env.simulator.import_object(v)
            # v.set_position(camera_pos)

            if not run_filters(env, objs_of_interest):
                continue

            rgb, segmask = env.simulator.renderer.render(("rgb", "seg"))

            rgb_img = Image.fromarray(np.uint8(rgb[:, :, :3] * 255))
            rgb_img.save(os.path.join(OUTPUT_DIR, f"{total_image_count}_rgb.png"))
            segmask = Image.fromarray(np.uint8(segmask[..., 0] * 255))
            segmask.save(os.path.join(OUTPUT_DIR, f"{total_image_count}_seg.png"))

            perturbation_image_count += 1
            total_image_count += 1

    while True:
        env.step(None)

    s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
