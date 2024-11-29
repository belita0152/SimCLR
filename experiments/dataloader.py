import numpy as np
from ..dataset.augmentation import SignalAugmentation as SigAug


def batch_dataloader(paths, batch_size, augmentations):
    augmentation = SigAug()
    np.random.shuffle(paths)

