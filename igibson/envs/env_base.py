import gym

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots.ant_robot import Ant
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.robots.fetch_gripper_robot import FetchGripper
from igibson.robots.fetch_robot import Fetch
from igibson.robots.freight_robot import Freight
from igibson.robots.humanoid_robot import Humanoid
from igibson.robots.husky_robot import Husky
from igibson.robots.jr2_kinova_robot import JR2_Kinova
from igibson.robots.jr2_robot import JR2
from igibson.robots.locobot_robot import Locobot
from igibson.robots.turtlebot_robot import Turtlebot
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.scenes.stadium_scene import StadiumScene
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config


class BaseEnv(gym.Env):
    """
    Base Env class, follows OpenAI Gym interface
    Handles loading scene and robot
    Functions like reset and step are not implemented
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        rendering_settings=None,
        device_idx=0,
        use_pb_gui=False,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless or gui mode
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param rendering_settings: rendering_settings to override the default one
        :param device_idx: device_idx: which GPU to run the simulation and rendering on
        :param use_pb_gui: concurrently display the interactive pybullet gui (for debugging)
        """
        self.config = parse_config(config_file)
        if scene_id is not None:
            self.config["scene_id"] = scene_id

        self.mode = mode
        self.action_timestep = action_timestep
        self.physics_timestep = physics_timestep
        self.rendering_settings = rendering_settings
        self.texture_randomization_freq = self.config.get("texture_randomization_freq", None)
        self.object_randomization_freq = self.config.get("object_randomization_freq", None)
        self.object_randomization_idx = 0
        self.num_object_randomization_idx = 10

        enable_shadow = self.config.get("enable_shadow", False)
        enable_pbr = self.config.get("enable_pbr", True)
        texture_scale = self.config.get("texture_scale", 1.0)

        if self.rendering_settings is None:
            # TODO: We currently only support the optimized renderer due to some issues with obj highlighting
            self.rendering_settings = MeshRendererSettings(
                enable_shadow=enable_shadow,
                enable_pbr=enable_pbr,
                msaa=False,
                texture_scale=texture_scale,
                optimized=True,
            )

        if mode == "vr":
            self.simulator = SimulatorVR(
                physics_timestep=physics_timestep,
                render_timestep=action_timestep,
                image_width=self.config.get("image_width", 128),
                image_height=self.config.get("image_height", 128),
                vertical_fov=self.config.get("vertical_fov", 90),
                device_idx=device_idx,
                rendering_settings=self.rendering_settings,
                vr_settings=VrSettings(use_vr=True),
                use_pb_gui=use_pb_gui,
            )
        else:
            self.simulator = Simulator(
                mode=mode,
                physics_timestep=physics_timestep,
                render_timestep=action_timestep,
                image_width=self.config.get("image_width", 128),
                image_height=self.config.get("image_height", 128),
                vertical_fov=self.config.get("vertical_fov", 90),
                device_idx=device_idx,
                rendering_settings=self.rendering_settings,
                use_pb_gui=use_pb_gui,
            )
        self.load()

    def reload(self, config_file):
        """
        Reload another config file
        Thhis allows one to change the configuration on the fly

        :param config_file: new config file path
        """
        self.config = parse_config(config_file)
        self.simulator.reload()
        self.load()

    def reload_model(self, scene_id):
        """
        Reload another scene model
        This allows one to change the scene on the fly

        :param scene_id: new scene_id
        """
        self.config["scene_id"] = scene_id
        self.simulator.reload()
        self.load()

    def reload_model_object_randomization(self):
        """
        Reload the same model, with the next object randomization random seed
        """
        if self.object_randomization_freq is None:
            return
        self.object_randomization_idx = (self.object_randomization_idx + 1) % (self.num_object_randomization_idx)
        self.simulator.reload()
        self.load()

    def get_next_scene_random_seed(self):
        """
        Get the next scene random seed
        """
        if self.object_randomization_freq is None:
            return None
        return self.scene_random_seeds[self.scene_random_seed_idx]

    def load(self):
        """
        Load the scene and robot
        """
        if self.config["scene"] == "empty":
            scene = EmptyScene()
            self.simulator.import_scene(
                scene, load_texture=self.config.get("load_texture", True), render_floor_plane=True
            )
        elif self.config["scene"] == "stadium":
            scene = StadiumScene()
            self.simulator.import_scene(scene, load_texture=self.config.get("load_texture", True))
        elif self.config["scene"] == "gibson":
            scene = StaticIndoorScene(
                self.config["scene_id"],
                waypoint_resolution=self.config.get("waypoint_resolution", 0.2),
                num_waypoints=self.config.get("num_waypoints", 10),
                build_graph=self.config.get("build_graph", False),
                trav_map_resolution=self.config.get("trav_map_resolution", 0.1),
                trav_map_erosion=self.config.get("trav_map_erosion", 2),
                pybullet_load_texture=self.config.get("pybullet_load_texture", False),
            )
            self.simulator.import_scene(scene, load_texture=self.config.get("load_texture", True))
        elif self.config["scene"] == "igibson":
            urdf_file = self.config.get("urdf_file", None)
            if urdf_file is None and not self.config.get("online_sampling", True):
                urdf_file = "{}_task_{}_{}_{}_fixed_furniture".format(
                    self.config["scene_id"],
                    self.config["task"],
                    self.config["task_id"],
                    self.config["instance_id"],
                )
            scene = InteractiveIndoorScene(
                self.config["scene_id"],
                urdf_file=urdf_file,
                waypoint_resolution=self.config.get("waypoint_resolution", 0.2),
                num_waypoints=self.config.get("num_waypoints", 10),
                build_graph=self.config.get("build_graph", False),
                trav_map_resolution=self.config.get("trav_map_resolution", 0.1),
                trav_map_erosion=self.config.get("trav_map_erosion", 2),
                trav_map_type=self.config.get("trav_map_type", "with_obj"),
                pybullet_load_texture=self.config.get("pybullet_load_texture", False),
                texture_randomization=self.texture_randomization_freq is not None,
                object_randomization=self.object_randomization_freq is not None,
                object_randomization_idx=self.object_randomization_idx,
                should_open_all_doors=self.config.get("should_open_all_doors", False),
                load_object_categories=self.config.get("load_object_categories", None),
                not_load_object_categories=self.config.get("not_load_object_categories", None),
                load_room_types=self.config.get("load_room_types", None),
                load_room_instances=self.config.get("load_room_instances", None),
                merge_fixed_links=self.config.get("merge_fixed_links", True)
                and not self.config.get("online_sampling", False),
            )
            # TODO: Unify the function import_scene and take out of the if-else clauses
            first_n = self.config.get("_set_first_n_objects", -1)
            if first_n != -1:
                scene._set_first_n_objects(first_n)
            self.simulator.import_ig_scene(scene)

        if self.config["robot"] == "Turtlebot":
            robot = Turtlebot(self.config)
        elif self.config["robot"] == "Husky":
            robot = Husky(self.config)
        elif self.config["robot"] == "Ant":
            robot = Ant(self.config)
        elif self.config["robot"] == "Humanoid":
            robot = Humanoid(self.config)
        elif self.config["robot"] == "JR2":
            robot = JR2(self.config)
        elif self.config["robot"] == "JR2_Kinova":
            robot = JR2_Kinova(self.config)
        elif self.config["robot"] == "Freight":
            robot = Freight(self.config)
        elif self.config["robot"] == "Fetch":
            robot = Fetch(self.config)
        elif self.config["robot"] == "Locobot":
            robot = Locobot(self.config)
        elif self.config["robot"] == "BehaviorRobot":
            robot = BehaviorRobot(self.simulator)
        elif self.config["robot"] == "FetchGripper":
            robot = FetchGripper(self.simulator, self.config)
        else:
            raise Exception("unknown robot type: {}".format(self.config["robot"]))

        if isinstance(robot, BehaviorRobot):
            self.simulator.import_behavior_robot(robot)
        else:
            self.simulator.import_robot(robot)

        self.scene = self.simulator.scene
        self.robots = self.simulator.robots

    def clean(self):
        """
        Clean up
        """
        if self.simulator is not None:
            self.simulator.disconnect()

    def close(self):
        """
        Synonymous function with clean
        """
        self.clean()

    def simulator_step(self):
        """
        Step the simulation.
        This is different from environment step that returns the next
        observation, reward, done, info.
        """
        self.simulator.step()

    def step(self, action):
        """
        Overwritten by subclasses
        """
        return NotImplementedError()

    def reset(self):
        """
        Overwritten by subclasses
        """
        return NotImplementedError()

    def set_mode(self, mode):
        """
        Set simulator mode
        """
        self.simulator.mode = mode
