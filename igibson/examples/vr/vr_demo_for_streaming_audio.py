""" This is a VR demo in a simple scene consisting of some objects to interact with, and space to move around.
Can be used to verify everything is working in VR, and iterate on current VR designs.
"""
import logging
import os

import pybullet as p
import pybullet_data

import igibson
from igibson.objects.articulated_object import ArticulatedObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots import BehaviorRobot, Turtlebot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.audio.ig_acoustic_mesh import getIgAcousticMesh
from igibson.audio.audio_system import AudioSystem
from igibson.agents.av_nav.ppo.ppo_trainer import PPOTrainer
from igibson.agents.av_nav.utils.environment import AVNavRLEnv
from igibson.objects.visual_marker import VisualMarker
import torch

# HDR files for PBR rendering
from igibson.simulator_vr import SimulatorVR
from igibson.utils.utils import parse_config


hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")


def main(selection="user", headless=False, short_exec=False):
    # VR rendering settings
    vr_rendering_settings = MeshRendererSettings(
        optimized=True,
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
    #s = SimulatorVR(mode="vr", rendering_settings=vr_rendering_settings, vr_settings=VrSettings(use_vr=True))

    #scene = InteractiveIndoorScene(
    #    "Rs_int", load_object_categories=["walls", "floors", "ceilings"], load_room_types=["kitchen"]
    #)


    bvr_config = os.path.join(igibson.configs_path, "behavior_robot_vr_behavior_task.yaml")
    exp_config = "C:/Users/capri28/Documents/iGibson-dev/igibson/examples/vr/audiogoal_continuous.yaml"
    # ppo_trainer = PPOTrainer(exp_config)
    # ppo_trainer.device = (
    #        torch.device("cuda", ppo_trainer.config['TORCH_GPU_ID'])
    #        if torch.cuda.is_available()
    #        else torch.device("cpu")
    #    )
    env = AVNavRLEnv(config_file=exp_config, mode='vr', scene_splits=['Rs_int'], rendering_settings=vr_rendering_settings, vr_settings=VrSettings(use_vr=True))
    #bvr_robot = env.robots[0]
    
    # ppo_trainer._setup_actor_critic_agent(env.observation_space, action_space=env.action_space)
    # ckpt_dict = ppo_trainer.load_checkpoint(r"C:\Users\Takara\Repositories\iGibson\igibson\agents\av_nav\data\audiogoal_continuous\checkpoints\ckpt.307.pth", map_location="cpu")
    # ppo_trainer.agent.load_state_dict(ckpt_dict["state_dict"])

    config = parse_config(bvr_config)
    bvr_robot = BehaviorRobot(**config["robot"])
    env.simulator.import_object(bvr_robot)
    bvr_robot.set_position_orientation([0.5, 0, 0.7], [0, 0, 0, 1])
    env.simulator.switch_main_vr_robot(bvr_robot)
    # env.simulator.main_vr_robot = None


    #rnn_hidden_states = torch.zeros(
    #        ppo_trainer.agent.actor_critic.net.num_recurrent_layers,
    #        1,
    #        ppo_trainer.config['hidden_size'],
    #        device=ppo_trainer.agent.device,
    #        )

    #prev_actions = torch.zeros(
    #        1, env.action_space.shape[0], device=ppo_trainer.agent.device, dtype=torch.long
    #    )
    #not_done_masks = torch.zeros(
    #        1, 1, device=ppo_trainer.agent.device
    #    )


    # source_obj = VisualMarker(
    #     visual_shape=p.GEOM_CYLINDER,
    #     rgba_color=[0, 0, 1, 0.3],
    #     radius=0.1,
    #     length= 0.2,
    #     initial_offset=[0, 0, 0.2 / 2.0],
    # )

    # env.simulator.import_object(source_obj)

    # The visual object indicating the target location may be visible
    # for instance in source_obj.renderer_instances:
    #     instance.hidden = True


    # Main simulation loop
    obs = env.reset()
    # source_pos = env.simulator.get_vr_pos()
    # source_obj.set_position(source_pos)
    env.audio_system.registerSource(
                bvr_robot._parts["eye"].head_visual_marker.get_body_ids()[0],
                "",
                enabled=True,
                repeat=True
            )
    while True:
        #bvr_robot.apply_action(env.simulator.gen_vr_robot_action())
        # obs = env.step(env.simulator.gen_vr_robot_action())
        env.step()
        # source_obj.set_position(env.simulator.get_vr_pos())
        #with torch.no_grad():
            #_, action, _, rnn_hidden_states = ppo_trainer.agent.actor_critic.act(
            #    obs,
            #    rnn_hidden_states,
            #    prev_actions,
            #    not_done_masks
            #)
        #obs = env.step(action)
        #prev_actions.copy_(action[0].tolist())

        # End demo by pressing overlay toggle
        env.audio_system.step()
        bvr_robot.apply_action(env.simulator.gen_vr_robot_action())

        if env.simulator.query_vr_event("left_controller", "overlay_toggle"):
            break

    env.simulator.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
