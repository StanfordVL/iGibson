import numpy as np
import os
from collections import OrderedDict
from contextlib import contextmanager

import pybullet as p
import pybullet_data
import gibson2
from gibson2.core.physics.interactive_objects import InteractiveObj, YCBObject, Object, VisualMarker
import gibson2.external.pybullet_tools.transformations as T

from gibson2.envs.kitchen.camera import Camera
import gibson2.envs.kitchen.env_utils as EU
import gibson2.external.pybullet_tools.utils as PBU
import gibson2.envs.kitchen.plan_utils as PU
import gibson2.envs.kitchen.skills as skills
from gibson2.envs.kitchen.objects import Faucet, Box, CoffeeMachine, MessyPlate, Hook, Tube, CoffeeGrinder
from gibson2.envs.kitchen.base_env import BaseEnv, EnvSkillWrapper


class TableTop(BaseEnv):
    def __init__(self, **kwargs):
        kwargs["robot_base_pose"] = ([0.5, 0.3, 1.2], [0, 0, 1, 0])
        super(TableTop, self).__init__(**kwargs)

    def _create_sensors(self):
        # PBU.set_camera(0, -45, 0.8, (0.5, -0.3, 1.0))
        PBU.set_camera(0, -45, 0.8, (0.0, -0.3, 1.0))
        self.camera = Camera(
            height=self._camera_width,
            width=self._camera_height,
            fov=60,
            near=0.01,
            far=10.,
            renderer=p.ER_TINY_RENDERER
        )
        self.camera.set_pose_ypr((0.0, -0.3, 1.0), distance=0.8, yaw=0, pitch=-45)

    def _create_fixtures(self):
        p.loadMJCF(os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml"))
        table_id = p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "table/table.urdf"),
            useFixedBase=True,
            basePosition=(0, 0, 0.0)
        )
        table = Object()
        table.loaded = True
        table.body_id = table_id
        self.fixtures.add_object("table", table)

