"""This module contains vr utility functions."""

import numpy as np
from gibson2.utils.utils import normalizeListVec

def move_player_no_body(s, rTouchX, rTouchY, movement_speed, relative_device):
    """Moves the VR player when they are not using a VR body. Takes in the simulator,
    information from the right touchpad, player movement speed and the device relative to which
    we would like to move."""
    curr_offset = s.get_vr_offset()
    right, _, forward = s.get_device_coordinate_system(relative_device)
    new_offset = translate_vr_position_by_vecs(rTouchX, rTouchY, right, forward, curr_offset, movement_speed)
    s.set_vr_offset(new_offset)

def get_normalized_translation_vec(right_frac, forward_frac, right, forward):
    """Generates a normalized translation vector that is a linear combination of forward and right."""
    vr_offset_vec = [right[i] * right_frac + forward[i] * forward_frac for i in range(3)]
    vr_offset_vec[2] = 0
    return normalizeListVec(vr_offset_vec)

def translate_vr_position_by_vecs(right_frac, forward_frac, right, forward, curr_offset, movement_speed):
    """Generates a normalized translation vector that is a linear combination of forward and right, the"""
    """direction vectors of the chosen VR device (HMD/controller), and adds this vector to the current offset."""
    vr_offset_vec = get_normalized_translation_vec(right_frac, forward_frac, right, forward)
    return [curr_offset[i] + vr_offset_vec[i] * movement_speed for i in range(3)]
