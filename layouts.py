"""Skeleton layouts (joint topology) for each pose source.

mediapipe : 33 MediaPipe Pose joints, optionally + 2 virtual joints (mid-hip, neck)
ntu       : 25 Kinect v2 joints of the NTU RGB+D dataset
"""
from dataclasses import dataclass, field

import numpy as np

MP_NAMES = [
    "nose", "l_eye_in", "l_eye", "l_eye_out", "r_eye_in", "r_eye", "r_eye_out",
    "l_ear", "r_ear", "mouth_l", "mouth_r", "l_shoulder", "r_shoulder", "l_elbow",
    "r_elbow", "l_wrist", "r_wrist", "l_pinky", "r_pinky", "l_index", "r_index",
    "l_thumb", "r_thumb", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
    "l_heel", "r_heel", "l_foot", "r_foot",
]
MP_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]
MP_FLIP = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
           (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
# parent of every joint, used for bone vectors (limbs point away from the body)
_MP_LIMB_PARENTS = {1: 0, 4: 0, 9: 0, 10: 0, 2: 1, 3: 2, 7: 3, 5: 4, 6: 5, 8: 6,
                    13: 11, 15: 13, 17: 15, 19: 15, 21: 15,
                    14: 12, 16: 14, 18: 16, 20: 16, 22: 16,
                    25: 23, 27: 25, 29: 27, 31: 27, 26: 24, 28: 26, 30: 28, 32: 28}

NTU_NAMES = [
    "base_spine", "mid_spine", "neck", "head", "l_shoulder", "l_elbow", "l_wrist",
    "l_hand", "r_shoulder", "r_elbow", "r_wrist", "r_hand", "l_hip", "l_knee",
    "l_ankle", "l_foot", "r_hip", "r_knee", "r_ankle", "r_foot", "spine_shoulder",
    "l_hand_tip", "l_thumb", "r_hand_tip", "r_thumb",
]
# (child, parent), 1-based as in the ST-GCN / 2s-AGCN code; 21 = spine_shoulder is the root
_NTU_PAIRS_1B = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7),
                 (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                 (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8),
                 (24, 25), (25, 12)]
NTU_EDGES = [(c - 1, p - 1) for c, p in _NTU_PAIRS_1B]
NTU_FLIP = [(4, 8), (5, 9), (6, 10), (7, 11), (12, 16), (13, 17), (14, 18), (15, 19),
            (21, 23), (22, 24)]


@dataclass
class Layout:
    name: str
    raw_joints: int                 # joints stored in the .npy files
    names: list
    edges: list                     # graph edges over all nodes (raw + virtual)
    center: int                     # root node for spatial partitioning
    parents: list                   # parent per node for bone vectors (root = itself)
    flip_pairs: list
    origin_joints: list             # averaged to find the body centre (raw indices)
    top_joints: list                # averaged to find the top of the torso
    torso_dims: int                 # 2 = measure torso in x,y only; 3 = x,y,z
    y_down: bool                    # image coordinates (True) or camera metres (False)
    virtual: list = field(default_factory=list)   # each = raw joints averaged into a new node

    @property
    def num_nodes(self):
        return self.raw_joints + len(self.virtual)

    def flip_index(self):
        idx = np.arange(self.num_nodes)
        for a, b in self.flip_pairs:
            idx[a], idx[b] = b, a
        return idx


def get_layout(name: str, virtual: bool = True) -> Layout:
    if name == "mediapipe" and virtual:
        MID_HIP, NECK = 33, 34
        parents = list(range(35))
        for c, p in _MP_LIMB_PARENTS.items():
            parents[c] = p
        parents[0] = NECK
        parents[11] = parents[12] = NECK
        parents[23] = parents[24] = MID_HIP
        parents[NECK] = MID_HIP
        parents[MID_HIP] = MID_HIP
        return Layout(
            name="mediapipe", raw_joints=33, names=MP_NAMES + ["mid_hip", "neck"],
            edges=MP_EDGES + [(MID_HIP, 23), (MID_HIP, 24), (MID_HIP, NECK),
                              (NECK, 11), (NECK, 12), (NECK, 0), (0, 9), (0, 10)],
            center=MID_HIP, parents=parents, flip_pairs=MP_FLIP,
            origin_joints=[23, 24], top_joints=[11, 12], torso_dims=2, y_down=True,
            virtual=[[23, 24], [11, 12]])
    if name == "mediapipe":
        # no virtual joints: connect head to shoulders directly, nose becomes the centre
        parents = list(range(33))
        for c, p in _MP_LIMB_PARENTS.items():
            parents[c] = p
        parents[11] = parents[12] = 0
        parents[23], parents[24] = 11, 12
        return Layout(
            name="mediapipe", raw_joints=33, names=list(MP_NAMES),
            edges=MP_EDGES + [(0, 11), (0, 12), (0, 9), (0, 10)],
            center=0, parents=parents, flip_pairs=MP_FLIP,
            origin_joints=[23, 24], top_joints=[11, 12], torso_dims=2, y_down=True)
    if name == "ntu":
        parents = list(range(25))
        for c, p in NTU_EDGES:
            parents[c] = p
        return Layout(
            name="ntu", raw_joints=25, names=list(NTU_NAMES), edges=list(NTU_EDGES),
            center=20, parents=parents, flip_pairs=NTU_FLIP,
            origin_joints=[0], top_joints=[20], torso_dims=3, y_down=False)
    raise ValueError(f"unknown layout {name!r}")
