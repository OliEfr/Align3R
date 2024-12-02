# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# global alignment optimization wrapper function
# --------------------------------------------------------
from enum import Enum

from PIL.ImageOps import scale
from matplotlib.scale import scale_factory
from wandb.wandb_torch import torch

from .optimizer import PointCloudOptimizer
from .modular_optimizer import ModularPointCloudOptimizer
from .pair_viewer import PairViewer
from ..viz import pts3d_to_trimesh


class GlobalAlignerMode(Enum):
    PointCloudOptimizer = "PointCloudOptimizer"
    ModularPointCloudOptimizer = "ModularPointCloudOptimizer"
    PairViewer = "PairViewer"

import torch.nn.functional as F

def global_aligner(dust3r_output, if_use_mono, mono_depths, device, mode=GlobalAlignerMode.PointCloudOptimizer, **optim_kw):
    # extract all inputs
    view1, view2, pred1, pred2 = [dust3r_output[k] for k in 'view1 view2 pred1 pred2'.split()]

    # build the optimizer
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        net = PointCloudOptimizer(view1, view2, pred1, pred2, if_use_mono, mono_depths, **optim_kw).to(device)
    elif mode == GlobalAlignerMode.ModularPointCloudOptimizer:
        net = ModularPointCloudOptimizer(view1, view2, pred1, pred2, **optim_kw).to(device)
    elif mode == GlobalAlignerMode.PairViewer:
        net = PairViewer(view1, view2, pred1, pred2, **optim_kw).to(device)
    else:
        raise NotImplementedError(f'Unknown mode {mode}')

    return net
