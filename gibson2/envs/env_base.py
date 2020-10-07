from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.robots.husky_robot import Husky
from gibson2.robots.ant_robot import Ant
from gibson2.robots.humanoid_robot import Humanoid
from gibson2.robots.jr2_robot import JR2
from gibson2.robots.jr2_kinova_robot import JR2_Kinova
from gibson2.robots.freight_robot import Freight
from gibson2.robots.fetch_robot import Fetch
from gibson2.robots.locobot_robot import Locobot
from gibson2.simulator import Simulator
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.scenes.stadium_scene import StadiumScene
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
import gym


class BaseEnv(gym.Env):
    '''
    a basic environment, step, observation and reward not implemented
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
        settings = MeshRendererSettings()
        #TODO(fxia22): pass from config file
        self.simulator = Simulator(mode=mode,
                                   physics_timestep=physics_timestep,
                                   render_timestep=action_timestep,
                                   image_width=self.config.get('image_width', 128),
                                   image_height=self.config.get('image_height', 128),
                                   vertical_fov=self.config.get('vertical_fov', 90),
                                   device_idx=device_idx,
                                   render_to_tensor=render_to_tensor,
                                   rendering_settings=settings,
                                   auto_sync=False)
        self.load()

    def reload(self, config_file):
        """
        Reload another config file, this allows one to change the environment on the fly

        :param config_file: new config file path
        """
        self.config = parse_config(config_file)
        self.simulator.reload()
        self.load()

    def reload_model(self, scene_id):
        """
        Reload another model, this allows one to change the environment on the fly
        :param scene_id: new scene_id
        """
        self.config['scene_id'] = scene_id
        self.simulator.reload()
        self.load()

    def load(self):
        """
        Load the scene and robot
        """
        if self.config['scene'] == 'empty':
            scene = EmptyScene()
            self.simulator.import_scene(scene, load_texture=self.config.get('load_texture', True))
        elif self.config['scene'] == 'stadium':
            scene = StadiumScene()
            self.simulator.import_scene(scene, load_texture=self.config.get('load_texture', True))
        elif self.config['scene'] == 'gibson':
            scene = StaticIndoorScene(
                self.config['scene_id'],
                waypoint_resolution=self.config.get('waypoint_resolution', 0.2),
                num_waypoints=self.config.get('num_waypoints', 10),
                build_graph=self.config.get('build_graph', False),
                trav_map_resolution=self.config.get('trav_map_resolution', 0.1),
                trav_map_erosion=self.config.get('trav_map_erosion', 2),
                pybullet_load_texture=self.config.get('pybullet_load_texture', False),
            )
            self.simulator.import_scene(scene, load_texture=self.config.get('load_texture', True))
        elif self.config['scene'] == 'igibson':
            scene = InteractiveIndoorScene(
                self.config['scene_id'],
                waypoint_resolution=self.config.get('waypoint_resolution', 0.2),
                num_waypoints=self.config.get('num_waypoints', 10),
                build_graph=self.config.get('build_graph', False),
                trav_map_resolution=self.config.get('trav_map_resolution', 0.1),
                trav_map_erosion=self.config.get('trav_map_erosion', 2),
                pybullet_load_texture=self.config.get('pybullet_load_texture', False),
            )
            #TODO: Unify the function import_scene and take out of the if-else clauses
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
            raise Exception('unknown robot type: {}'.format(self.config['robot']))

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

    def simulator_step(self):
        """
        Step the simulation, this is different from environment step where one can get observation and reward
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
        self.simulator.mode = mode
