import pybullet as p
import numpy as np
from PIL import Image, ImageDraw


def capture_raw(height, width, **kwargs):
    """
    use flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX to get
    depth and semantic segmentation map
    :param height: height of the image
    :param width: width of the image
    :param flags: flags to use, default 0
    :return: a tuple of images
    """
    return p.getCameraImage(width, height, **kwargs)


def get_segmentation_mask_object_and_link_index(seg_image):
    """
    Following example from
    https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/segmask_linkindex.py
    :param seg_image: [H, W] segmentation bitmap
    :return: object id map
    :return: link id map
    """
    assert(seg_image.ndim == 2)
    bmask = seg_image >= 0
    obj_idmap = bmask.copy().astype(np.int64) - 1
    link_idmap = obj_idmap.copy() - 1
    obj_idmap[bmask] = seg_image[bmask] & ((1 << 24) - 1)
    link_idmap[bmask] = (seg_image[bmask] >> 24) - 1
    return obj_idmap, link_idmap


def get_depth_map(depth_image, near=0.01, far=100.):
    """
    compute a depth map given a depth image and projection frustrum
    https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
    :param depth_image:
    :param near: frustrum near range
    :param far: frustrum far range
    :return: a depth map
    """
    assert (depth_image.ndim == 2)
    depth = far * near / (far - (far - near) * depth_image)
    return depth


def get_images(height, width, flags, **kwargs):
    ims = capture_raw(height, width, flags, **kwargs)
    obj_idmap, link_idmap = get_segmentation_mask_object_and_link_index(ims[4])
    depth_map = get_depth_map(ims[3])
    return ims[2], obj_idmap, link_idmap, depth_map


class Camera(object):
    def __init__(self, height, width, fov=60, near=0.01, far=100., renderer=p.ER_TINY_RENDERER):

        aspect = float(width) / float(height)
        self._height = height
        self._width = width
        self._near = near
        self._far = far
        self._view_matrix = p.computeViewMatrix([0, 0, 1], [0, 0, 0], [1, 0, 0])
        self._projection_matrix = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=near, farVal=far)
        self._renderer = renderer

    def set_pose(self, camera_pos, target_pos, up_vector):
        assert(len(camera_pos) == 3)
        assert(len(target_pos) == 3)
        assert(len(up_vector) == 3)
        self._view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)

    def set_pose_ypr(self, target_pos, distance, yaw, pitch, roll=0, up_axis=2):
        assert(len(target_pos) == 3)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target_pos,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=up_axis
        )

    def capture_raw(self):
        width, height, rgb, depth, seg = p.getCameraImage(
            self._width,
            self._height,
            list(self._view_matrix),
            list(self._projection_matrix),
            renderer=p.ER_TINY_RENDERER
        )
        assert(width == self._width)
        assert(height == self._height)
        rgb = np.reshape(rgb, (height, width, 4))
        depth = np.reshape(depth, (height, width))
        seg = np.reshape(seg, (height, width))
        return rgb, depth, seg

    def capture_frame(self):
        rgb, depth, seg = self.capture_raw()
        obj_idmap, link_idmap = get_segmentation_mask_object_and_link_index(seg)
        depth_map = get_depth_map(depth, near=self._near, far=self._far)
        return rgb[:, :, :3].astype(np.uint8), depth_map, obj_idmap, link_idmap

    def get_crops(self, object_ids, expand_ratio=1.1):
        rgb, depth, obj_seg, link_seg = self.capture_frame()
        bbox = get_bbox2d_from_segmentation(obj_seg, object_ids)
        return crop_pad_resize(rgb, bbox[:, 1:], 24, expand_ratio=expand_ratio)


def get_bbox2d_from_mask(bmask):
    """
    Get 2d bbox from a binary segmentation mask
    :param bmask: binary segmentation mask
    :return:
    """
    box = np.zeros(4, dtype=np.int64)
    coords_r, coords_c = np.where(bmask > 0)
    if len(coords_r) == 0:
        print('WARNING: empty bbox')
        return box
    box[0] = coords_r.min()
    box[1] = coords_c.min()
    box[2] = coords_r.max()
    box[3] = coords_c.max()
    return box


def union_boxes(boxes):
    """
    Union a list of boxes
    :param boxes: [N, 4] boxes
    :return:
    """
    assert(isinstance(boxes, (tuple, list)))
    boxes = np.vstack(boxes)
    new_box = boxes[0].copy()
    new_box[0] = boxes[:, 0].min()
    new_box[1] = boxes[:, 1].min()
    new_box[2] = boxes[:, 2].max()
    new_box[3] = boxes[:, 3].max()
    return new_box


def get_bbox2d_from_segmentation(seg_map, object_ids):
    """
    Get 2D bbox from a semantic segmentation map
    :param seg_map:
    :return:
    """
    all_bboxes = np.zeros([len(object_ids), 5], dtype=np.int64)
    for i in range(len(object_ids)):
        all_bboxes[i, 0] = object_ids[i]
        all_bboxes[i, 1:] = get_bbox2d_from_mask(seg_map == object_ids[i])
    return all_bboxes


def box_rc_to_xy(box):
    """
    box coordinate from (r1, c1, r2, c2) to (x1, y1, x2, y2)
    :param box: a
    :return: box
    """
    return np.array([box[1], box[0], box[3], box[2]], dtype=box.dtype)


def draw_boxes(image, boxes, labels=None):
    if labels is not None:
        assert(len(labels) == len(boxes))
    image = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(image)
    for b in boxes:
        draw.rectangle(box_rc_to_xy(b).tolist(), outline='green')
    return np.array(image)


def crop_pad_resize(images, bbox, target_size, expand_ratio=1.0):
    crops = np.zeros((bbox.shape[0], target_size, target_size, 3), dtype=images.dtype)
    im_pil = Image.fromarray(images.copy())
    for i, box in enumerate(bbox):
        if np.all(box == 0):
            continue
        box_center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        box_size = [(box[2] - box[0]), box[3] - box[1]]
        s = max(box_size) / 2 * expand_ratio
        new_box = [box_center[1] - s, box_center[0] - s, box_center[1] + s,  box_center[0] + s]
        crop = im_pil.crop(new_box).resize((target_size, target_size), resample=Image.BILINEAR)
        crops[i] = np.array(crop)
    return crops