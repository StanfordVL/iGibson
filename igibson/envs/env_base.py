from igibson.robots.turtlebot_robot import Turtlebot
from igibson.robots.husky_robot import Husky
from igibson.robots.ant_robot import Ant
from igibson.robots.humanoid_robot import Humanoid
from igibson.robots.jr2_robot import JR2
from igibson.robots.jr2_kinova_robot import JR2_Kinova
from igibson.robots.freight_robot import Freight
from igibson.robots.fetch_robot import Fetch
from igibson.robots.locobot_robot import Locobot
from igibson.simulator import Simulator
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.stadium_scene import StadiumScene
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.utils.utils import parse_config
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import gym


class BaseEnv(gym.Env):
    '''
    Base Env class, follows OpenAI Gym interface
    Handles loading scene and robot
    Functions like reset and step are not implemented
    '''

    def __init__(self,
                 config_file,
                 scene_id=None,
                 mode='headless',
                 action_timestep=1 / 10.0,
                 physics_timestep=1 / 240.0,
                 render_to_tensor=False,
                 device_idx=0):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless or gui mode
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: device_idx: which GPU to run the simulation and rendering on
        """
        self.config = parse_config(config_file)
        if scene_id is not None:
            self.config['scene_id'] = scene_id

        self.mode = mode
        self.action_timestep = action_timestep
        self.physics_timestep = physics_timestep
        self.texture_randomization_freq = self.config.get(
            'texture_randomization_freq', None)
        self.object_randomization_freq = self.config.get(
            'object_randomization_freq', None)
        self.object_randomization_idx = 0
        self.num_object_randomization_idx = 10

        enable_shadow = self.config.get('enable_shadow', False)
        enable_pbr = self.config.get('enable_pbr', True)
        texture_scale = self.config.get('texture_scale', 1.0)

        settings = MeshRendererSettings(enable_shadow=enable_shadow,
                                        enable_pbr=enable_pbr,
                                        msaa=False,
                                        texture_scale=texture_scale)

        self.simulator = Simulator(mode=mode,
                                   physics_timestep=physics_timestep,
                                   render_timestep=action_timestep,
                                   image_width=self.config.get(
                                       'image_width', 128),
                                   image_height=self.config.get(
                                       'image_height', 128),
                                   vertical_fov=self.config.get(
                                       'vertical_fov', 90),
                                   device_idx=device_idx,
                                   render_to_tensor=render_to_tensor,
                                   rendering_settings=settings)
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
        self.config['scene_id'] = scene_id
        self.simulator.reload()
        self.load()

    def reload_model_object_randomization(self):
        """
        Reload the same model, with the next object randomization random seed
        """
        if self.object_randomization_freq is None:
            return
        self.object_randomization_idx = (self.object_randomization_idx + 1) % \
            (self.num_object_randomization_idx)
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
        if self.config['scene'] == 'empty':
            scene = EmptyScene()
            self.simulator.import_scene(
                scene, load_texture=self.config.get('load_texture', True))
        elif self.config['scene'] == 'stadium':
            scene = StadiumScene()
            self.simulator.import_scene(
                scene, load_texture=self.config.get('load_texture', True))
        elif self.config['scene'] == 'gibson':
            scene = StaticIndoorScene(
                self.config['scene_id'],
                waypoint_resolution=self.config.get(
                    'waypoint_resolution', 0.2),
                num_waypoints=self.config.get('num_waypoints', 10),
                build_graph=self.config.get('build_graph', False),
                trav_map_resolution=self.config.get(
                    'trav_map_resolution', 0.1),
                trav_map_erosion=self.config.get('trav_map_erosion', 2),
                pybullet_load_texture=self.config.get(
                    'pybullet_load_texture', False),
            )
            self.simulator.import_scene(
                scene, load_texture=self.config.get('load_texture', True))
        elif self.config['scene'] == 'igibson':
            scene = InteractiveIndoorScene(
                self.config['scene_id'],
                waypoint_resolution=self.config.get(
                    'waypoint_resolution', 0.2),
                num_waypoints=self.config.get('num_waypoints', 10),
                build_graph=self.config.get('build_graph', False),
                trav_map_resolution=self.config.get(
                    'trav_map_resolution', 0.1),
                trav_map_erosion=self.config.get('trav_map_erosion', 2),
                trav_map_type=self.config.get('trav_map_type', 'with_obj'),
                pybullet_load_texture=self.config.get(
                    'pybullet_load_texture', False),
                texture_randomization=self.texture_randomization_freq is not None,
                object_randomization=self.object_randomization_freq is not None,
                object_randomization_idx=self.object_randomization_idx,
                should_open_all_doors=self.config.get(
                    'should_open_all_doors', False),
                load_object_categories=self.config.get(
                    'load_object_categories', None),
                load_room_types=self.config.get('load_room_types', None),
                load_room_instances=self.config.get(
                    'load_room_instances', None),
            )
            # TODO: Unify the function import_scene and take out of the if-else clauses
            first_n = self.config.get('_set_first_n_objects', -1)
            if first_n != -1:
                scene._set_first_n_objects(first_n)
            self.simulator.import_ig_scene(scene)

        if self.config['robot'] == 'Turtlebot':
            robot = Turtlebot(self.config)
        elif self.config['robot'] == 'Husky':
            robot = Husky(self.config)
        elif self.config['robot'] == 'Ant':
            robot = Ant(self.config)
        elif self.config['robot'] == 'Humanoid':
            robot = Humanoid(self.config)
        elif self.config['robot'] == 'JR2':
            robot = JR2(self.config)
        elif self.config['robot'] == 'JR2_Kinova':
            robot = JR2_Kinova(self.config)
        elif self.config['robot'] == 'Freight':
            robot = Freight(self.config)
        elif self.config['robot'] == 'Fetch':
            robot = Fetch(self.config)
        elif self.config['robot'] == 'Locobot':
            robot = Locobot(self.config)
        else:
            raise Exception(
                'unknown robot type: {}'.format(self.config['robot']))

        self.scene = scene
        self.robots = [robot]
        for robot in self.robots:
            self.simulator.import_robot(robot)

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
