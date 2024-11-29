import torch
import numpy as np
import torch.nn as nn


def get_backbone_model(model_name, parameters):
    from backbone import EEGNet

    if model_name == 'EEGNet':
        backbone = EEGNet(**parameters)
        return backbone


