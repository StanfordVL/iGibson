import numpy as np
import torch

from igibson.task_gen.constants import CATEGORY_FREQUENCY_MAPPING, CATEGORY_TOKEN_ID_MAPPING, MAX_DIMS, MIN_DIMS, PAD


def prepare_data(scene_graph):
    """Convert to an autoregressive sequence prediction task.
    - normalize and discretize to [0,255]
    - one-hot categorical tokens
    - autoregressive structure
    - factor across COLD attributes
    N=len(scene_graph)
    """
    # sort by category frequency
    # get category frequencies/size

    def format_graph_to_sequence(graph):
        """Order the nodes (objects).
        Returns a list of graph.values() in order by category frequency
        """
        obj_insts = list(sorted(graph.keys()))

        def category_frequency(idx):
            category = graph["category"]
            freq = CATEGORY_FREQUENCY_MAPPING[category]
            return freq

        order = sorted(range(len(obj_insts)), key=category_frequency)
        return [graph[obj_insts[idx]] for idx in order]

    def format_cat(cat):
        """Convert to token_ids"""
        token = CATEGORY_TOKEN_ID_MAPPING[cat]
        return token

    def format_pos(pos):
        """Normalize and discretize to [0,255]"""
        offset = np.zeros(3)  # TODO offset
        pos = (pos - offset) / (MAX_DIMS - MIN_DIMS)  # normalize
        pos = (pos * 255.0).astype(np.int32)  # discretize
        return pos

    def format_orn(orn):
        """Normalize and discretize to [0,359]"""
        orn = (orn * 255.0).astype(np.int32)  # discretize
        return orn

    def format_bbox(bbox):
        """Normalize and discretize to [0,255]? check SceneFormer code"""
        offset = np.zeros(3)  # TODO offset
        bbox = (bbox - offset) / (MAX_DIMS - MIN_DIMS)  # normalize
        bbox = (bbox * 255.0).astype(np.int32)  # discretize
        return bbox

    def to_numpy(lst, shape=None, dtype=None):
        return np.array(lst, shape=shape, dtype=dtype)  # TODO check

    seq = format_graph_to_sequence(scene_graph)
    cat_seq = list(map(lambda _: format_cat(_["cat"]), seq))  # shape=(B,N,N_CAT)
    pos_seq = list(map(lambda _: format_pos(_["pos"]), seq))  # shape=(B,N,3)
    orn_seq = list(map(lambda _: format_orn(_["orn"]), seq))  # shape=(B,N,1)
    bbox_seq = list(map(lambda _: format_bbox(_["bbox"]), seq))  # shape=(B,N,3,2)

    # stride
    orn_seq = [PAD] + orn_seq
    pos_seq = [PAD, PAD] + pos_seq
    bbox_seq = [PAD, PAD, PAD] + bbox_seq

    # convert to np arrs
    cat_seq = to_numpy(cat_seq, dtype=np.int32)
    pos_seq = to_numpy(pos_seq, dtype=np.int32)
    orn_seq = to_numpy(orn_seq, dtype=np.int32)
    bbox_seq = to_numpy(bbox_seq, dtype=np.int32)

    # stack/concat
    tokens = np.concatenate(
        [  # TODO np.expand_dims? or reshape bbox? check bbox shape
            cat_seq,  # [C]ategory
            orn_seq,  # [O]rientation
            pos_seq,  # [L]ocation
            bbox_seq,  # [D]imension
        ],
        axis=0,
    )  # TODO # shape=(B,N,)

    tokens = torch.from_numpy(tokens)
    tokens = tokens.float()

    return tokens

    # TODO padding to account for different scene_graph sizes aka sequence lengths (huggingface?)
