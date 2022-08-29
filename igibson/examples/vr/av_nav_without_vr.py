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
import numpy as np
import cv2
# HDR files for PBR rendering
from igibson.simulator_vr import SimulatorVR
from igibson.utils.utils import parse_config
import imageio
from scipy.io import wavfile
import librosa
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import transforms3d as tf3d

def main(selection="user", headless=False, short_exec=False):
    # VR rendering settings

    exp_config = "C:/Users/capri28/Documents/iGibson-dev/igibson/examples/vr/audiogoal_continuous.yaml"
    ppo_trainer = PPOTrainer(exp_config)
    ppo_trainer.device = (
           torch.device("cuda", ppo_trainer.config['TORCH_GPU_ID'])
           if torch.cuda.is_available()
           else torch.device("cpu")
       )
    env = AVNavRLEnv(config_file=exp_config, mode='headless', scene_splits=['Pomaria_1_int'])
    # p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # objects = [
    #     ("table/table.urdf", (1.00000, 0., 0.000000), (0.000000, 0.000000, 0.0, 1.0)),
    # ]

    # for item in objects:
    #     fpath = item[0]
    #     pos = item[1]
    #     orn = item[2]
    #     item_ob = ArticulatedObject(fpath, scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    #     env.simulator.import_object(item_ob)
    #     item_ob.set_position(pos)
    #     item_ob.set_orientation(orn)

    print(env.task)
    # source_obj = VisualMarker(
    #     visual_shape=p.GEOM_CYLINDER,
    #     rgba_color=[0, 0, 0, 1.],
    #     radius=0.01,
    #     length= 0.2,
    #     initial_offset=[0, 0, 0.2 / 2.0],
    # )

    # env.simulator.import_object(source_obj)

    # # The visual object indicating the target location may be visible
    # for instance in source_obj.renderer_instances:
    #     instance.hidden = True


    # # Main simulation loop
    # source_obj.set_position([0.5, 0., 0.725])
    # env.audio_system.registerSource(
    #             source_obj.get_body_ids()[0],
    #             "telephone.wav",
    #             enabled=True,
    #         )

    env.reset()
    delta = tf3d.quaternions.axangle2quat([0, 0, 1], 0)
    start = np.array([1.0, 0., 0., 0.])
    final = tf3d.quaternions.qmult(start, delta)
    final = [final[1], final[2],final[3],final[0]]
    print("final", final)
    env.robots[0].set_position_orientation([0., 0.5, 0.1], final)

    
    ppo_trainer._setup_actor_critic_agent(env.observation_space, action_space=env.action_space)
    # ckpt_dict = ppo_trainer.load_checkpoint(r"C:\Users\capri28\Documents\iGibson-dev\igibson\examples\vr\data\audiogoal_continuous\checkpoints\ckpt.307.pth", map_location="cpu")
    # ppo_trainer.agent.load_state_dict(ckpt_dict["state_dict"])

    # bvr_robot.set_position_orientation([0.5, 0, 0], [0, 0, 0, 1])

    rnn_hidden_states = torch.zeros(
           ppo_trainer.agent.actor_critic.net.num_recurrent_layers,
           1,
           ppo_trainer.config['hidden_size'],
           device=ppo_trainer.agent.device,
           )

    prev_actions = torch.zeros(
           1, env.action_space.shape[0], device=ppo_trainer.agent.device, dtype=torch.long
       )

    not_done_masks = torch.zeros(
           1, 1, device=ppo_trainer.agent.device
       )

    
   
    rgb = []
    td_map = []
    prev_control = np.zeros((2,))
    while True:
        try:
            state, _,_,_ = env.step(np.array([-0.1,.1]))
            print(env.robots[0].get_position_orientation())
            j_control, ctl_type = env.robots[0]._actions_to_control(np.array([0.5, 1.0]))
            lin_vel = env.robots[0].wheel_radius * (j_control[0] + j_control[1]) / 2
            ang_vel = env.robots[0].wheel_radius * (j_control[1] - j_control[0]) / (env.robots[0].wheel_axle_length)
            MOMENTUM = 0.2
            LIN_SCALE = 0.8
            ANG_SCALE = 0.8
            vel_msg = MOMENTUM * prev_control[0] + (1 - MOMENTUM) * lin_vel * LIN_SCALE
            ang_msg = MOMENTUM * prev_control[1] + (1 - MOMENTUM) * ang_vel * ANG_SCALE
            prev_control = [vel_msg, ang_msg]
            # print([prev_control])

            # print(env.robots[0].get_linear_velocity(), env.robots[0].get_angular_velocity())
            depth_map = state["depth_video"].squeeze() * 255.0
            if not isinstance(depth_map, np.ndarray):
                depth_map = depth_map.cpu().numpy()

            depth_map = depth_map.astype(np.uint8)
            depth_map = np.stack([depth_map for _ in range(3)], axis=2)
            rgb.append(depth_map)

            topdown = state["top_down_video"]
            if not isinstance(topdown, np.ndarray):
                topdown = topdown.cpu().numpy()
            # print(topdown.shape)
            td_map.append(topdown)
            # source_obj.set_position(env.simulator.get_vr_pos())
            # with torch.no_grad():
                #_, action, _, rnn_hidden_states = ppo_trainer.agent.actor_critic.act(
                #    obs,
                #    rnn_hidden_states,
                #    prev_actions,
                #    not_done_masks
                #)
            #obs = env.step(action)
            #prev_actions.copy_(action[0].tolist())
        except KeyboardInterrupt:
            break
    env.audio_system.disconnect()
    env.simulator.disconnect()

    writer = imageio.get_writer(
        "video.mp4",
        fps=10,
        quality=5,
    )
    for im in rgb:
        writer.append_data(im)
    writer.close()

    writer = imageio.get_writer(
        "td_video.mp4",
        fps=10,
        quality=8,
    )
    for im in td_map:
        writer.append_data(im)
    writer.close()

def get_spectrogram(audio_buffer):
    def compute_stft(signal):
        n_fft = 512
        hop_length = 160
        win_length = 400
        stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
        return stft

    spectrogram_per_channel = []
    for mono_idx in range(2):
        spectrogram_per_channel.append(np.log1p(compute_stft(audio_buffer[:, mono_idx])))

    spectrogram = np.stack(spectrogram_per_channel, axis=-1)
    return spectrogram

def plot_spectrogram(spec, fname="spectrogram.png"):
    import librosa.display
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spec, y_axis='log', x_axis='time', ax=ax, sr=44100, hop_length=160)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.savefig(fname)

def plot_wave_form(audio_buffer, fname="wave_form.png"):
    fig, ax = plt.subplots()
    audio_buffer = np.array(audio_buffer)
    x_lim = np.arange(0, len(audio_buffer)) / 44100
    ax.set_title('Audio Waveform')
    plt.plot(x_lim, audio_buffer)
    plt.savefig(fname)
    plt.close()

def plot_audio():
    samplerate, data = wavfile.read('mixed_noise_tele.wav')
    
    print(data.shape)
    data = np.array(data, dtype=np.float32, order='C') / 32768.0
    # data = data[:44100*3,:]
    print(np.amin(data), np.amax(data))
    spec = get_spectrogram(data)
    print(spec.shape)
    plot_spectrogram(spec[:, :, 0], "l_spec.png")
    plot_spectrogram(spec[:, :, 1], "r_spec.png")
    plot_wave_form(data, fname="l_wave_form.png")
    # plot_wave_form(data[:, 1], fname="r_wave_form.png")

def concat_audios():
    audios = []
    for i in range(14):
        samplerate, data = wavfile.read('noise/test_' + str(i) + '.wav')
        print(i)
        audios.append(data)
    merged = np.concatenate(audios, axis=0)
    print(merged.shape)
    wavfile.write("background_noise.wav", 44100, merged)
    



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
    plot_audio()
    # concat_audios()
