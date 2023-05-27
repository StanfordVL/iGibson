#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange

def dilate_tensor(x, size, iterations=1):
    """
    x - (bs, C, H, W)
    size - int / tuple of intes

    Assumes a kernel of ones with size 'size'.
    """
    if type(size) == int:
        padding = size // 2
    else:
        padding = tuple([v // 2 for v in size])
    for i in range(iterations):
        x = F.max_pool2d(x, size, stride=1, padding=padding)

    return x


def ground_projection(img_feats, spatial_locs, valid_inputs, local_shape, K, eps=-1e16):
    r"""Inputs:
        img_feats       - (bs, F, H/K, W/K) image features to project to ground plane
        spatial_locs    - (bs, 2, H, W)
                          for each batch, H and W, the (x, y) locations on map are given.
        valid_inputs    - (bs, 1, H, W) ByteTensor
        local_shape     - (outh, outw) tuple indicating size of output projection
        K               - image_size / map_shape ratio (needed for sampling values from spatial_locs)
        eps             - fill_value
    Outputs:
        proj_feats      - (bs, F, s, s)
    """
    device = img_feats.device
    outh, outw = local_shape
    bs, F, HbyK, WbyK = img_feats.shape
    img_feat_locs = (
        (torch.arange(0, HbyK, 1) * K + K / 2).long().to(device),
        (torch.arange(0, WbyK, 1) * K + K / 2).long().to(device),
    )

    input_feats = img_feats
    input_idxes = spatial_locs[
        :, :, img_feat_locs[0][:, None], img_feat_locs[1]
    ]  # (bs, 2, HbyK, WbyK)
    valid_inputs_depth = valid_inputs[
        :, :, img_feat_locs[0][:, None], img_feat_locs[1]
    ]  # (bs, 1, HbyK, WbyK)
    valid_inputs_depth = valid_inputs_depth.squeeze(1)  # (bs, HbyK, WbyK)
    invalid_inputs_depth = ~valid_inputs_depth

    output_feats = torch.zeros(bs, F, outh, outw).to(device)
    output_feats.fill_(eps)
    output_feats_rshp = output_feats.view(*output_feats.shape[:2], -1)
    input_idxes_flip = torch.flip(input_idxes, [1])  # convert x, y to y, x

    invalid_writes = (
        (input_idxes_flip[:, 0] >= outh)
        | (input_idxes_flip[:, 1] >= outw)
        | (input_idxes_flip[:, 0] < 0)
        | (input_idxes_flip[:, 1] < 0)
        | invalid_inputs_depth
    )  # (bs, H, W)

    # Set the idxes for all invalid locations to (0, 0)
    input_idxes_flip[:, 0][invalid_writes] = 0
    input_idxes_flip[:, 1][invalid_writes] = 0

    invalid_writes = invalid_writes.float().unsqueeze(1)

    input_feats_masked = input_feats * (1 - invalid_writes) + eps * invalid_writes
    input_feats_rshp = input_feats_masked.view(bs, F, -1)
    input_idxes_rshp = (
        input_idxes_flip[:, 0, :, :] * outw + input_idxes_flip[:, 1, :, :]
    )
    input_idxes_rshp = input_idxes_rshp.view(bs, 1, -1).expand(-1, F, -1)
    output_feats_rshp = torch.full((bs, 1, F * outh * outw), eps, device=input_feats_rshp.device)
    output_feats_rshp = torch_scatter.scatter(
        input_feats_rshp, input_idxes_rshp, dim=2, out=output_feats_rshp, dim_size=outh * outw, reduce = 'max', #fill_value=eps,
    )
    output_feats = output_feats_rshp.view(bs, F, outh, outw)
    eps_mask = (output_feats == eps).float()
    output_feats = output_feats * (1 - eps_mask) + eps_mask * (output_feats - eps)

    return output_feats


def compute_spatial_locs(
    depth_inputs,
    local_shape,
    local_scale,
    camera_params,
    min_depth=0.0,
    truncate_depth=-1.0,
    return_height=False,
):
    """
    Inputs:
        depth_inputs  - (bs, 1, imh, imw) depth values per pixel in `units`.
        local_shape   - (s, s) tuple of ground projection size
        local_scale   - cell size of ground projection in `units`
        camera_params - (fx, fy, cx, cy) tuple
    Outputs:
        spatial_locs  - (bs, 2, imh, imw) x,y locations of projection per pixel
        valid_inputs  - (bs, 1, imh, imw) ByteTensor (all locations where
                        depth measurements are available)

    Conventions for the map: The agent is standing at the bottom center of the map and facing upward
    in the egocentric coordinate.
    """
    fx, fy, cx, cy = camera_params
    bs, _, imh, imw = depth_inputs.shape
    s = local_shape[1]
    device = depth_inputs.device
    # Precompute the projection values
    # 2D image coordinates
    x = rearrange(torch.arange(0, imw), "w -> () () () w")
    y = rearrange(torch.arange(imh, 0, step=-1), "h -> () () h ()")
    x, y = x.float().to(device), y.float().to(device)
    xx = (x - cx) / fx
    yy = (y - cy) / fy

    # 3D real-world coordinates (in meters)
    Z = depth_inputs # (1, 128, 128)
    X = xx * Z
    # print(yy)
    # print(Z[:,:,:5])
    Y = yy * Z
    # print(Z.shape)
    # print(yy * Z[:,:,:,:2])
    valid_inputs = abs(depth_inputs - min_depth) > 0.8
    if truncate_depth > 0:
        valid_inputs = valid_inputs & (depth_inputs <= truncate_depth)
    # 2D ground projection coordinates (in meters)
    # Note: map_scale - dimension of each grid in meters
    # - depth/scale + (s-1) since image convention is image y downward
    # and agent is facing upwards.
    z_gp = (-(Z / local_scale) + (s - 1)).round().long()  # (bs, 1, imh, imw)
    x_gp = ((X / local_scale) + (s - 1) / 2).round().long()  # (bs, 1, imh, imw)

    if return_height:
        outputs = (torch.cat([x_gp, z_gp], dim=1), valid_inputs, Y)
    else:
        outputs = (torch.cat([x_gp, z_gp], dim=1), valid_inputs)

    return outputs


class MapNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._local_map_shape = cfg["local_map_shape"]
        self.map_scale = cfg["map_scale"]  # meters
        self.config = cfg
        self.min_depth = cfg["min_depth"]
        self.max_depth = cfg["max_depth"]

        # Camera params for depth projection
        self.hfov = math.radians(cfg["hfov"])
        self.vfov = math.radians(cfg["vfov"])

    def forward(self, img_feats, depth):
        """
        Inputs:
            img_feats - (bs, F, h, w)
            depth - (bs, 3, H, W)
        """
        # Based on the image size, set the camera parameters
        H, W = depth.shape[2:4]
        self.fx = (W / 2) * (1 / math.tan(self.hfov / 2))
        self.fy = (H / 2) * (1 / math.tan(self.vfov / 2))
        self.cx = W / 2
        self.cy = H / 2
        self.K = depth.shape[2] / img_feats.shape[2]
        assert self.K == depth.shape[3] / img_feats.shape[3]
        depth = self._process_depth(depth)
        spatial_locs, valid_inputs = self._compute_spatial_locs(depth)
        x_gp = self._ground_projection(img_feats, spatial_locs, valid_inputs)

        return x_gp

    def _ground_projection(self, img_feats, spatial_locs, valid_inputs, eps=-1e16):
        """
        Inputs:
            img_feats       - (bs, F, H/K, W/K)
            spatial_locs    - (bs, 2, H, W)
                              for each batch, H and W, the (x, y) locations on map are given.
            valid_inputs    - (bs, 1, H, W) ByteTensor
            eps             - fill_value
        Outputs:
            proj_feats      - (bs, F, s, s)
        """
        output_feats = ground_projection(
            img_feats,
            spatial_locs,
            valid_inputs,
            self._local_map_shape[1:],
            self.K,
            eps=eps,
        )
        return output_feats

    def _compute_spatial_locs(
        self, depth_inputs, return_height=False, truncate_depth=-1.0
    ):
        """
        Inputs:
            depth_inputs - (bs, 1, imh, imw) depth values per pixel in meters.
        Outputs:
            spatial_locs - (bs, 2, imh, imw) x,y locations of projection per
                           pixel
            valid_inputs - (bs, 1, imh, imw) ByteTensor (all locations where
                           depth measurements are available)
        """
        camera_params = (self.fx, self.fy, self.cx, self.cy)
        local_scale = self.map_scale
        local_shape = self._local_map_shape[1:]
        outputs = compute_spatial_locs(
            depth_inputs,
            local_shape,
            local_scale,
            camera_params,
            min_depth=self.min_depth,
            truncate_depth=truncate_depth,
            return_height=return_height,
        )  # spatial_locs, valid_inputs, pixel_heights (optional)

        return outputs

    def _process_depth(self, depth):
        """
        Inputs:
            depth - (bs, 3, H, W)
        NOTE - this is specific to HabitatSimulator. The depth in meters is
        truncated to (self.min_depth, self.max_depth) meters, and scaled
        between 0 to 1. These operations have to be undone to generate the
        actual depth values in meters.
        """
        proc_depth = depth[:, 0].unsqueeze(1)
        proc_depth = proc_depth * self.max_depth + self.min_depth
        return proc_depth


class DepthProjectionNet(MapNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.camera_height = cfg["camera_height"]
        self.height_thresh = cfg["height_thresholds"]
        self.truncate_depth = cfg["truncate_depth"]
        self.K = 1

    def forward(self, depth):
        """
        Inputs:
            depth - (bs, 3, H, W)
        """
        bs, _, H, W = depth.shape
        device = depth.device
        height_thresh = self.height_thresh
        # Based on the image size, set the camera parameters
        self.fx = (W / 2) * (1 / math.tan(self.hfov / 2))
        self.fy = (H / 2) * (1 / math.tan(self.vfov / 2))
        self.cx = W / 2
        self.cy = H / 2
        # Preprocess depth and compute corresponding l
        depth = self._process_depth(depth)
        spatial_locs, valid_inputs, pixel_heights = self._compute_spatial_locs(
            depth, truncate_depth=self.truncate_depth, return_height=True
        )

        pixel_heights = pixel_heights + self.camera_height
        high_filter_mask = (pixel_heights < height_thresh[1]) & valid_inputs
        low_filter_mask = (pixel_heights > height_thresh[0]) & valid_inputs
        obstacle_mask = high_filter_mask & low_filter_mask
        free_mask = (pixel_heights < height_thresh[0]) & valid_inputs
        obstacle_mask = obstacle_mask.float()  # (bs, 1, H, W)
        free_mask = free_mask.float()  # (bs, 1, H, W)
        # Classify each pixel as obstacle / free-space
        x_gp_obstacles = self._ground_projection(
            obstacle_mask, spatial_locs, valid_inputs
        )  # (bs, 1, s, s)
        x_gp_free = self._ground_projection(
            free_mask, spatial_locs, valid_inputs
        )  # (bs, 1, s, s)
        # Dilate obstacles in x_gp
        x_gp_obstacles = dilate_tensor(x_gp_obstacles, 3, iterations=2)
        # Dilate free space in x_gp
        x_gp_free = dilate_tensor(x_gp_free, 5, iterations=2)
        # Compute explored space from free space and obstacles
        x_gp_explored = torch.max(x_gp_free, x_gp_obstacles)
        x_gp = torch.cat([x_gp_obstacles, x_gp_explored], dim=1)
        return x_gp


def main():
    from igibson.utils.utils import parse_config
    import cv2
    import imageio as iio
    import numpy as np
    config_pth = "../config/savi_occ/savi_rt_pretraining.yaml"
    config = parse_config(config_pth)
    proj = DepthProjectionNet(config['ego_proj'])
    depth = iio.v2.imread("depth_copy.png")
    depth = torch.tensor(depth).unsqueeze(0).unsqueeze(0) / 255.
    projed = proj(depth)
    z = torch.zeros((1, projed.shape[-1], projed.shape[-1]))
    projed = (torch.cat((projed.squeeze(0), z), 0).permute(1,2,0).cpu().numpy()* 255).astype(np.uint8) 
    # projed = ((1-projed).squeeze(0).permute(1,2,0).cpu().numpy()* 255).astype(np.uint8) 
    cv2.imwrite("projected_1.png", projed)

if __name__ == "__main__":
    main()
