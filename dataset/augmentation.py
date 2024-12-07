import torch
import random
import numpy as np


class SignalAugmentation(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.gn_scaling = np.arange(0.05, 0.1, 0.01)  # for 5% ~ 10% gaussian noise.
        self.n_permutation = 4

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
        # for: 각각의 input x (sample) 에 대해 노이즈를 추가할지 말지 결정
        # 조건: random.random() < p 이면 순열 적용
        # indexes = input_length 범위 안에서, n_permutation -1 개의 인덱스를 무작위로 선택.
        # indexes = 정렬된 상태로 변환
        # indexes = 리스트의 시작과 끝에 0과 input_length (데이터 조각의 길이)를 추가

        # for문 활용, 해당 indexes에 맞게 데이터를 segment -> samples 리스트에 segment들 저장

        # 2. Shuffle
        # 데이터 조각의 인덱스를 np.random.permutation
        # 순열된 인덱스에 따라 samples에서 데이터 조각 선택 -> nx에 추가

        # 3. Merge
        # np.concatenate 를 활용해 전체 합치기 (axis=-1)

        # else:  random.random() > p 에 해당한다면 순열 적용하지 않고, 바로 데이터 추가
        # aug_x.append(sample)


        # 데이터 반환
        # aug_x = np.array(aug_x)  # np.array 형태로 바꿔서 출력
        # return aug_x







        return self.x[..., self.n_permutation, :], self.y
