""" This is a VR demo in a simple scene consisting of some objects to interact with, and space to move around.
Can be used to verify everything is working in VR, and iterate on current VR designs.
"""
import logging
import os
from xml.dom.minidom import DocumentFragment

import pybullet as p
import pybullet_data
import random
import matplotlib
import igibson
from igibson.agents.av_nav.utils.utils import batch_obs
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

def global_to_local(pointgoal_global, pose, angle):
    delta = pointgoal_global - pose[:2]
    delta_theta = np.arctan2(delta[1], delta[0]) - angle
    d = np.linalg.norm(delta)
    return np.array([d * np.cos(delta_theta), d * np.sin(delta_theta)])

def local_to_global(pointgoal_local, pose, angle):
    d = np.linalg.norm(pointgoal_local)
    theta = np.arctan2(pointgoal_local[1], pointgoal_local[0])
    return np.array([pose[0] + d*np.cos(theta+angle), pose[1] + d * np.sin(theta+angle)])

def main(trail=0):
    # VR rendering settings
    path = "result/" + "av_nav" + "/" + str(trail) + "/"
    os.mkdir(path)
    exp_config = "C:/Users/capri28/Documents/iGibson-dev/igibson/examples/vr/data/audiogoal_continuous/config/audiogoal_continuous_semantic.yaml"
    ppo_trainer = PPOTrainer(exp_config)
    ppo_trainer.device = (
           torch.device("cuda", ppo_trainer.config['TORCH_GPU_ID'])
           if torch.cuda.is_available()
           else torch.device("cpu")
       )
    env = AVNavRLEnv(config_file=exp_config, mode='headless', scene_splits=['Wainscott_0_int'], device_idx=0,trail=trail)
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

    obs = env.reset()
    obs = batch_obs([obs], ppo_trainer.device)
    delta = tf3d.quaternions.axangle2quat([0, 0, 1], 0)
    start = np.array([1.0, 0., 0., 0.])
    final = tf3d.quaternions.qmult(start, delta)
    final = [final[1], final[2],final[3],final[0]]
    print("final", final)
    # env.robots[0].set_position_orientation([.0, .0, 0.1], final)

    
    ppo_trainer._setup_actor_critic_agent(env.observation_space, action_space=env.action_space)
    ckpt_dict = ppo_trainer.load_checkpoint(r"C:\Users\capri28\Documents\iGibson-dev\igibson\examples\vr\data\savi_task\av_nav\ckpt.83.pth", map_location="cpu")
    ppo_trainer.agent.load_state_dict(ckpt_dict["state_dict"])
    ppo_trainer.agent.to(ppo_trainer.device)

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
    depth = []
    td_map = []
    step = 0
    trajs_x = []
    trajs_y = []
    prev_control = np.zeros((2,))
    while True:
        try:
            with torch.no_grad():
                _, action, _, rnn_hidden_states = ppo_trainer.agent.actor_critic.act(
                   obs,
                   rnn_hidden_states,
                   prev_actions,
                   not_done_masks,
                   True
                )
            step += 1
            obs, reward, done, info= env.step(np.array(action[0].cpu().numpy()))
            # env.task.load_gt_rt_map(env)
            # print("rel location!!!!", obs['task_obs'], obs["pose_sensor"])
            # print("local", global_to_local(np.array([0.,-2.]),np.array([0.5,0.,0.]), 2.5 *np.pi))
            # print("global", local_to_global(np.array([-2.,0.5]),np.array([0.5,0.,0.]), 2.5 *np.pi))
            prev_actions.copy_(action[0])
            # print(reward, done)
            state = obs
            # command = env.robots[0]._controllers["base"]._preprocess_command(np.array([1.0,1.0]))#action[0])
            # print(env.robots[0]._controllers["base"].command_output_limits[1][0])
            # print("command", command)
            # right_wheel_vel = command[0]
            # left_wheel_vel = command[1]
            # ang_vel = (right_wheel_vel - left_wheel_vel) * env.robots[0].wheel_radius / (env.robots[0].wheel_axle_length)
            # lin_vel = (right_wheel_vel + left_wheel_vel) * env.robots[0].wheel_radius / 2

            # prev_control = [lin_vel, ang_vel]
            # print([prev_control])

            # print(env.robots[0].get_linear_velocity(), env.robots[0].get_angular_velocity())
            print(env.robots[0].get_position(), step)
            depth_map = state["depth_video"].squeeze() * 255.0
            if not isinstance(depth_map, np.ndarray):
                depth_map = depth_map.cpu().numpy()

            depth_map = depth_map.astype(np.uint8)
            depth_map = np.stack([depth_map for _ in range(3)], axis=2)

            rgb_im = state["rgb_video"].squeeze() * 255.0
            if not isinstance(rgb_im, np.ndarray):
                rgb_im = rgb_im.cpu().numpy()
            rgb_im = rgb_im.astype(np.uint8)
            rgb.append(rgb_im)
            depth.append(depth_map)

            topdown = state["top_down_video"]
            topdown = np.flip(topdown, axis=0)
            if not isinstance(topdown, np.ndarray):
                topdown = topdown.cpu().numpy()
            # print(topdown.shape)
            td_map.append(topdown)
            
            map_pos = env.scene.world_to_map(np.array(env.robots[0].get_position())[:2])
            if len(trajs_x) == 0:
                trajs_y.append(map_pos[0] * 1280/env.scene.trav_map_size)
                trajs_x.append(map_pos[1] * 1280/env.scene.trav_map_size)
            elif trajs_x[-1] != map_pos[0] and trajs_y[-1] != map_pos[1]:
                trajs_y.append(map_pos[0] * 1280/env.scene.trav_map_size)
                trajs_x.append(map_pos[1] * 1280/env.scene.trav_map_size)

            if done:
                break
            obs = batch_obs([obs], ppo_trainer.device)

        except KeyboardInterrupt:
            break
    # env.audio_system.disconnect()
    env.simulator.disconnect()

    writer = imageio.get_writer(
        path + "rgb_video.mp4",
        fps=10,
        quality=5,
    )
    for im in rgb:
        writer.append_data(im)
    writer.close()

    writer = imageio.get_writer(
        path + "depth_video.mp4",
        fps=10,
        quality=5,
    )
    for im in depth:
        writer.append_data(im)
    writer.close()
    writer = imageio.get_writer(
        path + "td_video.mp4",
        fps=10,
        quality=8,
    )
    for i in range(len(td_map)):
        topdown = td_map[i]
        trajs_x_np = np.array(trajs_x)[:i+1]
        trajs_y_np = np.array(trajs_y)[:i+1]
        if trajs_x_np.shape[0] >= 2 and trajs_x_np.shape[0] < 4:
            fig = plt.figure(figsize=(12.8,12.8))
            ax = fig.add_subplot()
            fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
            topdown = ax.imshow(topdown)
            ax.plot(trajs_x_np, trajs_y_np, 'r', linewidth=2)
            ax.axis('tight')
            ax.axis('off')
            # fig.add_axes(ax)
            fig.canvas.draw()
            topdown = np.array(fig.canvas.renderer._renderer)[:,:,:3]
            plt.close()
            # topdown[traj!=0] == traj
        elif trajs_x_np.shape[0] >= 4:
            fig = plt.figure(figsize=(12.8,12.8))
            ax = fig.add_subplot()
            fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
            topdown = ax.imshow(topdown)
            ax.plot(trajs_x_np, trajs_y_np,'r',  linewidth=2)
            ax.axis('tight')
            ax.axis('off')
            # fig.add_axes(ax)
            fig.canvas.draw()
            topdown = np.array(fig.canvas.renderer._renderer)[:,:,:3]
            plt.close()
            # topdown[traj!=0] = traj
        else:
            topdown[map_pos[0], map_pos[1]] == np.array([255,0,0])
        new_im = np.zeros_like(topdown)
        new_im[:,:,0] = topdown[:,:,2]
        new_im[:,:,1] = topdown[:,:,1]
        new_im[:,:,2] = topdown[:,:,0]
        writer.append_data(new_im)
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
    spec = get_spectrogram(data)
    print(spec.shape)
    plot_spectrogram(spec[:, :, 0], "l_spec.png")
    plot_spectrogram(spec[:, :, 1], "r_spec.png")
    plot_wave_form(data[:, 0], fname="l_wave_form.png")
    plot_wave_form(data[:, 1], fname="r_wave_form.png")
    print(np.amin(data), np.amax(data))

def concat_audios():
    audios = []
    for i in range(2):
        samplerate, data = wavfile.read('background_noise_' + str(i) + '.wav')
        print(i)
        audios.append(data)
    merged = np.concatenate(audios, axis=0)
    print(merged.shape)
    wavfile.write("bg_noise.wav", 44100, merged)
    



if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    matplotlib.use("Agg")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    for i in range(30):
        main(i)
    # concat_audios()
