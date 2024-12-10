import numpy as np
import os
import mne
from scipy.signal import butter, lfilter


class Dataset(object):
    """
    EEG Dataset - AD, CN, FTD
    전처리 하지 않은 raw file 사용
    https://openneuro.org/datasets/ds004504
    """

    def __init__(self):
        self.ch_names = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
            'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
        ]

        self.sfreq = 500
        self.labels = ['AD', 'CN', 'FTD']
        self.second = 10
        self.epoch_num = 10
        self.low, self.high = 0.5, 45

    def butter_bandpass_filter(self, data, order=5):
        nyq = 0.5 * self.sfreq
        low, high = self.low / nyq, self.high / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y

    def epoching(self, data, info):
        data = mne.io.RawArray(data, info=info)
        epoched = mne.make_fixed_length_epochs(raw=data, duration=10)
        epoch_len = epoched.get_data().shape[0]
        epoched_data = epoched.get_data()[
                       np.random.choice(epoch_len, self.epoch_num), :, :int(self.second * info['sfreq'])
                       ]
        return epoched_data

    def parser(self):
        data_list, info_list = [], []
        for sub_num in range(1, 6):
            # path를 parser -> fname으로 입력
            # OpenNeuro -> 전처리 하지 않은 raw file 사용
            fname = "../../GNN/AD-GNN/preproc/data/sub-0{}/eeg/sub-0{}_task-eyesclosed_eeg.set".format(f"{sub_num:02d}", f"{sub_num:02d}")
            raw = mne.io.read_raw_eeglab(fname, preload=False)

            # 1. slicing signal
            np.random.seed(10)
            info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='eeg')

            # 2. band pass filter (0.5 ~ 45 Hz)
            # 전처리 파일 사용한다면 pass, 전처리 하지 않은 raw file 사용한다면 실행
            data = self.butter_bandpass_filter(raw._data)

            # 3. epoching (random으로 n개의 epoch 추출: default 10개)
            epoched_data = self.epoching(data, info)
            data_list.append(epoched_data)
            info_list.append(info)

        return data_list, info_list

#
# if __name__ == '__main__':
#     dataset = Dataset()
#     data_list, info_list = dataset.parser()
#     print(data_list)