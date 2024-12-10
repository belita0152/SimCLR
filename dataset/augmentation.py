import random
import numpy as np


class SignalAugmentation(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.gn_scaling = np.arange(0.05, 0.1, 0.01)  # for 5% ~ 10% gaussian noise.
        self.n_permutation = 5
        self.second = 10
        self.sampling_rate = 500
        self.input_length = self.second * self.sampling_rate

    def random_gaussian_noise(self, p=0.5):
        """Randomly add Gaussian noise to all channels."""
        # 데이터 준비 #
        x = np.array(self.x)
        aug_x = []
        std = np.random.choice(self.gn_scaling, 1) * np.std(self.x)

        # for: 각각의 x sample에 대해 노이즈를 추가할지 말지 결정
        for sample in x:
            if random.random() < p:
                noise = np.random.normal(loc=0, scale=std, size=sample.shape)
                sample = sample + noise
                aug_x.append(sample)
            else:
                aug_x.append(sample)

        aug_x = np.array(aug_x)  # np.array 형태로 바꿔서 출력
        return aug_x

    def random_permutation(self, p=0.5):
        """Randomly segment, shuffle, and merge the signals."""
        # 데이터 준비 #
        x = np.array(self.x)
        aug_x = []

        # 1. Segment
        for sample in x:
            if random.random() < p:
                indexes = list(np.random.choice(self.input_length, self.n_permutation - 1, replace=False))
                indexes += [0, self.input_length]
                indexes = list(np.sort(indexes))

                # for문 활용, 해당 indexes에 맞게 데이터를 segment -> samples 리스트에 segment들 저장
                segments = []
                for idx_1, idx_2 in zip(indexes[:-1], indexes[1:]):
                    segments.append(sample[:, idx_1:idx_2])

                # 2. Shuffle
                # 데이터 조각의 인덱스를 np.random.permutation
                shuffled_segments = []
                for idx in np.random.permutation(np.arange(self.n_permutation)):
                    # 순열된 인덱스에 따라 samples에서 데이터 조각 선택 -> shuffled_segments에 추가
                    shuffled_segments.append(segments[idx])

                # 3. Merge
                # np.concatenate 를 활용해 전체 합치기 (axis=-1)
                shuffled_segments = np.concatenate(shuffled_segments, axis=-1)
                aug_x.append(shuffled_segments)

        # else:  random.random() > p 에 해당한다면 순열 적용하지 않고, 바로 데이터 추가
        else:
            aug_x.append(shuffled_segments)

        # 데이터 반환
        aug_x = np.array(aug_x)  # np.array 형태로 바꿔서 출력
        return aug_x









