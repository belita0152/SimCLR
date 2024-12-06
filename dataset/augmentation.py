import torch
import numpy as np


class SignalAugmentation(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.gn_scaling = np.arange(0.05, 0.1, 0.01)  # for 5% ~ 10% gaussian noise.

    def gaussian_noise(self, x, p=0.5):
        """Randomly add Gaussian noise to all channels."""
        ######## 데이터 준비 ########
        x = np.array(x)
        aug_x = []
        # gaussian noise scaling factor 지정하기 = 5% ~ 10% 까지 list 생성 -> init에서 정의
        # scaling factor에 따라 noise의 std 결정

        # for: 각각의 input x에 대해 노이즈를 추가할지 말지 결정
        # 노이즈 추가 조건: random.random() < p 이면 노이즈 추가
        ## noise = np.random.normal() 사용 / loc=0, scale=std, size=aug_x.shape
        ## 각각의 x = x + noise
        ## aug_x.append(x)

        # else:  # 즉 노이즈를 추가하지 않는 조건에서는
        # aug_x.append(x)  # 원본 데이터를 그대로 사용

        # aug_x = np.array(aug_x)  # np.array 형태로 바꿔서 출력
        # return aug_x

    def channels_permute(self, x, y, permutation):
        """Permute EEG channels according to fixed permutation matrix."""
        return x[..., permutation, :], y
