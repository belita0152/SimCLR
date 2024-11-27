import argparse
import mne
import numpy as np
import torch
import torchvision
import utils
import simclr
import os


class Dataset(object):
    """
    EEG Dataset - AD, CN, FTD
    https://openneuro.org/datasets/ds004504
    """

    def __init__(self):
        self.ch_names = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4',
            'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
            'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
        ]

        self.sfreq = 500
        self.labels = ['AD', 'CN', 'FTD']
        self.second = 4

    def parser(self, fname) -> ((np.array, np.array), mne.Info):  # fname -> Add to dictionary = 이런 이름이 있는 거야~ 라고 명명
        for sub_num in 88:
            # path를 parser -> fname으로 입력
            fname = "../GNN/AD-GNN/preproc/data/sub-0{}/eeg/sub-0{}_task-eyesclosed_eeg.set".format(f"{sub_num:02d}", f"{sub_num:02d}")
            x = mne.io.read_raw_eeglab(fname, preload=False)
            info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='eeg')
            raw = mne.EpochsArray(x, info=info)

        return raw, info


# if __name__ == '__main__':
#   args.device = torch.device('cuda' if args.cuda else 'cpu')
#   model = utils.model.get_model(args)
#
#   optimizer = torch.optim.Adam(
#       model.parameters(),
#       lr = 0.001,
#       weight_decay = 1e-4
#       )
#
#   dataloaders = {}
#
#   dataloaders['train'] = torch.utils.data.Dataloader(
#       TrainDataset(args),
#       batch_size = args.batch_size,
#       shuffle = True,
#       num_workers = args.num_workers
#       )
#
#   loss_fn = utils.ntxent.loss_function
#   simclrobj = simclr.SimCLR(model, optimizer, dataloaders, loss_fn)
#   simclrobj.train(args, 200, 10)








