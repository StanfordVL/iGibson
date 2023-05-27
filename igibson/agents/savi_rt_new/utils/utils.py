import glob
import os
from collections import defaultdict
from turtle import rt
from typing import Dict, List, Optional
import random
import numbers
import copy
import json
from gym.spaces import Box
import quaternion

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Size, Tensor
from igibson.agents.savi_rt_new.utils.tensorboard_utils import TensorboardWriter
from igibson.agents.savi_rt_new.utils.logs import logger

import imageio
import tqdm
from scipy.io import wavfile
import moviepy.editor as mpy
from moviepy.audio.AudioClip import CompositeAudioClip



class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1).contiguous()
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


class CategoricalNetWithMask(nn.Module):
    def __init__(self, num_inputs, num_outputs, masking):
        super().__init__()
        self.masking = masking

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, features, action_maps):
        probs = f.softmax(self.linear(features))
        if self.masking:
            probs = probs * torch.reshape(action_maps, (action_maps.shape[0], -1)).float()

        return CustomFixedCategorical(probs=probs)
    
    
    
class CustomNormal(torch.distributions.normal.Normal):
    def sample(
        self, sample_shape: Size = torch.Size()  # noqa: B008
    ) -> Tensor:
        return super().rsample(sample_shape)

    def log_probs(self, actions) -> Tensor:
        ret = super().log_prob(actions).sum(-1).unsqueeze(-1)
        return ret


class GaussianNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        min_log_std: int,
        max_log_std: int,
        use_log_std: bool,
        use_softplus: bool,
        action_activation: str
    ) -> None:
        super().__init__()

        self.action_activation = action_activation
        self.use_log_std = use_log_std
        self.use_softplus = use_softplus
        if use_log_std:
            self.min_std = min_log_std
            self.max_std = max_log_std
        else:
            self.min_std = min_std
            self.max_std = max_std

        self.mu = nn.Linear(num_inputs, num_outputs)
        self.std = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0)
        nn.init.orthogonal_(self.std.weight, gain=0.01)
        nn.init.constant_(self.std.bias, 0)

    def forward(self, x: Tensor) -> CustomNormal:
        mu = self.mu(x)
        if self.action_activation == "tanh":
            mu = torch.tanh(mu)

        std = torch.clamp(self.std(x), min=float(self.min_std), max=float(self.max_std))
        if self.use_log_std:
            std = torch.exp(std)
        if self.use_softplus:
            std = torch.nn.functional.softplus(std)

        return CustomNormal(mu, std)
    


def linear_decay(epoch: int, total_num_updates: int) -> float:
    r"""Returns a multiplicative factor for linear value decay
    Args:
        epoch: current epoch number
        total_num_updates: total number of epochs
    Returns:
        multiplicative factor that decreases param value linearly
    """
    return 1 - (epoch / float(total_num_updates))


def exponential_decay(epoch: int, total_num_updates: int, decay_lambda: float) -> float:
    r"""Returns a multiplicative factor for linear value decay
    Args:
        epoch: current epoch number
        total_num_updates: total number of epochs
        decay_lambda: decay lambda
    Returns:
        multiplicative factor that decreases param value linearly
    """
    return np.exp(-decay_lambda * (epoch / float(total_num_updates)))


def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def batch_obs(
    observations: List[Dict], device: Optional[torch.device] = None, skip_list = []
) -> Dict[str, torch.Tensor]:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.
    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None
    Returns:
        transposed dict of lists of observations.
    """
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            if sensor in skip_list:
                continue
            batch[sensor].append(to_tensor(obs[sensor]))

    for sensor in batch:
        batch[sensor] = torch.stack(batch[sensor], dim=0).to(
            device=device, dtype=torch.float
        )
        if sensor == "bump":
            batch["bump"] = batch["bump"][:, None]
        if sensor == "map_resolution":
            batch["map_resolution"] = batch["map_resolution"][:, None]
 
    return batch


def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int, eval_interval: int
) -> Optional[str]:
    r""" Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
    (sorted by time of last modification).
    Args:
        checkpoint_folder: directory to look for checkpoints.
        previous_ckpt_ind: index of checkpoint last returned.
        eval_interval: number of checkpoints between two evaluation
    Returns:
        return checkpoint path if (previous_ckpt_ind + 1)th checkpoint is found
        else return None.
    """
    assert os.path.isdir(checkpoint_folder), (
        f"invalid checkpoint folder " f"path {checkpoint_folder}"
    )
    models_paths = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/*"))
    )
    models_paths.sort(key=os.path.getmtime)
    ind = previous_ckpt_ind + eval_interval
    if ind < len(models_paths):
        return models_paths[ind]
    return None



class ResizeCenterCropper(nn.Module):
    def __init__(self, size, channels_last: bool = False):
        r"""An nn module the resizes and center crops your input.
        Args:
            size: A sequence (w, h) or int of the size you wish to resize/center_crop.
                    If int, assumes square crop
            channels_list: indicates if channels is the last dimension
        """
        super().__init__()
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        assert len(size) == 2, "forced input size must be len of 2 (w, h)"
        self._size = size
        self.channels_last = channels_last

    def transform_observation_space(
        self, observation_space, trans_keys=["rgb", "depth"]
    ):
        size = self._size
        observation_space = copy.deepcopy(observation_space)
        if size:
            for key in observation_space.spaces:
                if (
                    key in trans_keys
                    and observation_space.spaces[key].shape != size
                ):
                    logger.info(
                        "Overwriting CNN input size of %s: %s" % (key, size)
                    )
                    observation_space.spaces[key] = self.overwrite_gym_box_shape(
                        observation_space.spaces[key], size
                    )
        self.observation_space = observation_space
        return observation_space
    
    def overwrite_gym_box_shape(self, box: Box, shape) -> Box:
        if box.shape == shape:
            return box
        shape = list(shape) + list(box.shape[len(shape) :])
        low = box.low if np.isscalar(box.low) else np.min(box.low)
        high = box.high if np.isscalar(box.high) else np.max(box.high)
        return Box(low=low, high=high, shape=shape, dtype=box.dtype)

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._size is None:
            return input

        return center_crop(
            image_resize_shortest_edge(
                input, max(self._size), channels_last=self.channels_last
            ),
            self._size,
            channels_last=self.channels_last,
        )

def quaternion_from_coeff(coeffs: np.ndarray) -> np.quaternion:
    r"""Creates a quaternions from coeffs in [x, y, z, w] format"""
    quat = np.quaternion(0, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat    
    
def cartesian_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi

    
def image_resize_shortest_edge(
    img, size: int, channels_last: bool = False
) -> torch.Tensor:
    """Resizes an img so that the shortest side is length of size while
        preserving aspect ratio.

    Args:
        img: the array object that needs to be resized (HWC) or (NHWC)
        size: the size that you want the shortest edge to be resize to
        channels: a boolean that channel is the last dimension
    Returns:
        The resized array as a torch tensor.
    """
    img = to_tensor(img)
    no_batch_dim = len(img.shape) == 3
    if len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    if no_batch_dim:
        img = img.unsqueeze(0)  # Adds a batch dimension
    if channels_last:
        h, w = img.shape[-3:-1]
        if len(img.shape) == 4:
            # NHWC -> NCHW
            img = img.permute(0, 3, 1, 2).contiguous()
        else:
            # NDHWC -> NDCHW
            img = img.permute(0, 1, 4, 2, 3).contiguous()
    else:
        # ..HW
        h, w = img.shape[-2:]

    # Percentage resize
    scale = size / min(h, w)
    h = int(h * scale)
    w = int(w * scale)
    img = torch.nn.functional.interpolate(
        img.float(), size=(h, w), mode="area"
    ).to(dtype=img.dtype)
    if channels_last:
        if len(img.shape) == 4:
            # NCHW -> NHWC
            img = img.permute(0, 2, 3, 1).contiguous()
        else:
            # NDCHW -> NDHWC
            img = img.permute(0, 1, 3, 4, 2).contiguous()
    if no_batch_dim:
        img = img.squeeze(dim=0)  # Removes the batch dimension
    return img


def center_crop(img, size, channels_last: bool = False):
    """Performs a center crop on an image.

    Args:
        img: the array object that needs to be resized (either batched or unbatched)
        size: A sequence (w, h) or a python(int) that you want cropped
        channels_last: If the channels are the last dimension.
    Returns:
        the resized array
    """
    if channels_last:
        # NHWC
        h, w = img.shape[-3:-1]
    else:
        # NCHW
        h, w = img.shape[-2:]

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    assert len(size) == 2, "size should be (h,w) you wish to resize to"
    cropx, cropy = size

    startx = w // 2 - (cropx // 2)
    starty = h // 2 - (cropy // 2)
    if channels_last:
        return img[..., starty : starty + cropy, startx : startx + cropx, :]
    else:
        return img[..., starty : starty + cropy, startx : startx + cropx]    

    
    

def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().
    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().
    Returns:
        generated image of a single frame.
    """
    egocentric_view_l_rgb: List[np.ndarray] = []
    egocentric_view_l_depth: List[np.ndarray] = []
    egocentric_view_l_depth_proj: List[np.ndarray] = []
    top_down_view: List[np.ndarray] = []
    rt_map_view: List[np.ndarray] = []
    rt_map_gt_view: List[np.ndarray] = []

    if "rgb" in observation:
        rgb = observation["rgb_video"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view_l_rgb.append(rgb.astype(np.uint8))

    # draw depth map if observation has depth info
    if "depth" in observation:
        depth_map = observation["depth_video"].squeeze() * 255.0
        if not isinstance(depth_map, np.ndarray):
            depth_map = depth_map.cpu().numpy()

        depth_map = depth_map.astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        egocentric_view_l_depth.append(depth_map)

        depth_proj = observation["depth_proj"] * 255.0
        if not isinstance(depth_proj, np.ndarray):
            depth_proj = depth_proj.cpu().numpy()
        egocentric_view_l_depth_proj.append(depth_proj.astype(np.uint8))

    if "top_down" in observation:
        top_down = observation["top_down_video"]
        if not isinstance(top_down, np.ndarray):
            top_down = top_down.cpu().numpy()
            
        top_down_view.append(top_down.astype(np.uint8))

    # add image goal if observation has image_goal info
    # if "imagegoal" in observation:
    #     rgb = observation["imagegoal"]
    #     if not isinstance(rgb, np.ndarray):
    #         rgb = rgb.cpu().numpy()

    #     egocentric_view_l.append(rgb)

    if "rt_map" in observation:
        rt_map = observation["rt_map"]
        rt_map = to_tensor(rt_map)
        rt_preds = torch.where(torch.sigmoid(rt_map) >= 0.5, 1, 0).squeeze(0)
        if not isinstance(rt_preds, np.ndarray):
            rt_preds = rt_preds.cpu().numpy()
        pred = (rt_preds * 255).astype(np.uint8) 
        pred = np.stack([pred for _ in range(3)], axis=2)
        rt_map_view.append(pred)

        rt_map_gt = observation["rt_map_gt"]
        if not isinstance(rt_map_gt, np.ndarray):
            rt_map_gt = rt_map_gt.cpu().numpy()
        rt_map_gt = (rt_map_gt * 255).astype(np.uint8)
        rt_map_gt = np.stack([rt_map_gt for _ in range(3)], axis=2)
        rt_map_gt_view.append(rt_map_gt)

    egocentric_view_rgb = np.concatenate(egocentric_view_l_rgb, axis=1)
    egocentric_view_depth = np.concatenate(egocentric_view_l_depth, axis=1)
    egocentric_view_depth_proj = np.concatenate(egocentric_view_l_depth_proj, axis=1)
    top_down_view = np.concatenate(top_down_view, axis=1)
    rt_map_view = np.concatenate(rt_map_view, axis=1)
    rt_map_gt_view = np.concatenate(rt_map_gt_view, axis=1)

    return egocentric_view_rgb, egocentric_view_depth, egocentric_view_depth_proj, top_down_view, rt_map_view, rt_map_gt_view

def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
        **kwargs,
    )
    logger.info(f"Video created: {os.path.join(output_dir, video_name)}")
    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()

# def images_to_video_with_audio(
#     images: List[np.ndarray],
#     output_dir: str,
#     video_name: str,
#     audios: List[str],
#     sr: int,
#     fps: int = 1,
#     quality: Optional[float] = 5,
#     **kwargs
# ):
#     r"""Calls imageio to run FFMPEG on a list of images. For more info on
#     parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
#     Args:
#         images: The list of images. Images should be HxWx3 in RGB order.
#         output_dir: The folder to put the video in.
#         video_name: The name for the video.
#         audios: raw audio files
#         fps: Frames per second for the video. Not all values work with FFMPEG,
#             use at your own risk.
#         quality: Default is 5. Uses variable bit rate. Highest quality is 10,
#             lowest is 0.  Set to None to prevent variable bitrate flags to
#             FFMPEG so you can manually specify them using output_params
#             instead. Specifying a fixed bitrate using ‘bitrate’ disables
#             this parameter.
#     """
#     assert 0 <= quality <= 10
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
#     print("len images", len(images))
#     print("len audios", len(audios))
#     print("audios[0]", audios[0].shape)
#     assert len(images) == len(audios) * fps
#     audio_clips = []
#     temp_file_name = '/tmp/{}.wav'.format(random.randint(0, 10000))
#     # use amplitude scaling factor to reduce the volume of sounds
#     amplitude_scaling_factor = 100
#     for i, audio in enumerate(audios):
#         wavfile.write(temp_file_name, sr, audio.T / amplitude_scaling_factor)
#         audio_clip = mpy.AudioFileClip(temp_file_name)
#         audio_clip = audio_clip.set_duration(1)
#         audio_clip = audio_clip.set_start(i)
#         audio_clips.append(audio_clip)
#     composite_audio_clip = CompositeAudioClip(audio_clips)
#     video_clip = mpy.ImageSequenceClip(images, fps=fps)
#     video_with_new_audio = video_clip.set_audio(composite_audio_clip)
#     video_with_new_audio.write_videofile(os.path.join(output_dir, video_name))
#     os.remove(temp_file_name)

    
def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    scene_name: str,
    sound: str,
    sr: int,
    episode_id: int,
    checkpoint_idx: int,
    metric_name: str,
    metric_value: float,
    tb_writer: TensorboardWriter,
    fps: int = 10,
    audios: List[str] = None
) -> None:
    r"""Generate video according to specified information.
    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
        audios: raw audio files
    Returns:
        None
    """
    if len(images) < 1:
        return

    video_name = f"{scene_name}_{sound}_{metric_name}{metric_value:.2f}"
    if "disk" in video_option:
        assert video_dir is not None
        if audios is None:
            images_to_video(images, video_dir, video_name)
        else:
            images_to_video_with_audio(images, video_dir, video_name, audios, sr, fps=fps)
            
            

            
d3_40_colors_rgb: np.ndarray = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)