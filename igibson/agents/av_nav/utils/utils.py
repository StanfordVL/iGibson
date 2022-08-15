import glob
import os
from collections import defaultdict
from typing import Dict, List, Optional
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Size, Tensor
from igibson.agents.av_nav.utils.tensorboard_utils import TensorboardWriter
from igibson.agents.av_nav.utils.logs import logger

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
    observations: List[Dict], device: Optional[torch.device] = None
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
            batch[sensor].append(to_tensor(obs[sensor]))

    for sensor in batch:
        batch[sensor] = torch.stack(batch[sensor], dim=0).to(
            device=device, dtype=torch.float
        )
        if sensor == "bump":
            batch["bump"] = batch["bump"][:, None]

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



def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().
    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().
    Returns:
        generated image of a single frame.
    """
    egocentric_view_l: List[np.ndarray] = []
    egocentric_view_d: List[np.ndarray] = []
    topdown_view_l: List[np.ndarray] = []
    
    if "rgb" in observation:
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()
        egocentric_view_l.append(rgb)
    # draw depth map if observation has depth info

    if "depth" in observation:
        depth_map = observation["depth_video"].squeeze() * 255.0
        if not isinstance(depth_map, np.ndarray):
            depth_map = depth_map.cpu().numpy()

        depth_map = depth_map.astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        egocentric_view_d.append(depth_map)

    # add image goal if observation has image_goal info
    if "top_down" in observation:
        topdown = observation["top_down"]
        if not isinstance(topdown, np.ndarray):
            topdown = topdown.cpu().numpy()
        topdown_view_l.append(topdown)
    
    if "imagegoal" in observation:
        rgb = observation["imagegoal"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view_l.append(rgb)

    assert (
        len(egocentric_view_l) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view_l, axis=1)
    egocentric_view_d = np.concatenate(egocentric_view_d, axis=1)
    if "top_down" not in observation:
        topdown_view = egocentric_view
    else:
        topdown_view = np.concatenate(topdown_view_l, axis=1)
    frame = egocentric_view
    frame_topdown = topdown_view
    frame_depth = egocentric_view_d
    return frame, frame_depth, frame_topdown

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

    video_name = f"{scene_name}_{sound}_{episode_id}_{metric_name}_{metric_value:.2f}"
    if "disk" in video_option:
        assert video_dir is not None
        if audios is None:
            images_to_video(images, video_dir, video_name)
        else:
            images_to_video_with_audio(images, video_dir, video_name, audios, sr, fps=fps)