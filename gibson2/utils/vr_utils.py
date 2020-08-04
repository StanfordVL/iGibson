import numpy as np
from gibson2.utils.utils import normalizeListVec

def translate_vr_position_by_vecs(right_frac, forward_frac, right, forward, curr_offset, movement_speed):
    """Generates a normalized translation vector that is a linear combination of forward and right, the"""
    """direction vectors of the chosen VR device (HMD/controller), and adds this vector to the current offset."""
    vr_offset_vec = [right[i] * right_frac + forward[i] * forward_frac for i in range(3)]
    vr_offset_vec[2] = 0
    vr_offset_vec = normalizeListVec(vr_offset_vec)
    return [curr_offset[i] + vr_offset_vec[i] * movement_speed for i in range(3)]
