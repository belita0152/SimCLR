"""
Image/SimCLR => Base Encoder = ResNet
EEG/SimCLR => Base Encoder = EEGNet
"""
import torch.nn as nn


class EEGNet(nn.Module):  # Pytorch의 모든 NN은 nn.Module을 상속받음
    # https://github.com/vlawhern/arl-eegmodels
    def __init__(self, n_channels, n_classes, samples, dropout_rate, sampling_rate, F1, D):
        super(EEGNet, self).__init__()  # 부모 클래스의 초기화 호출

        # Block 1: Temporal Convolution + Depthwise Convolution
        kernel_length = int(sampling_rate // 2)

        # Temporal Convolution
        self.Conv2D = nn.Sequential(
            nn.Conv2d(1,
                      F1,
                      (1, kernel_length),
                      stride=1,
                      padding=(0, int(kernel_length // 2)),
                      bias=False),
            nn.BatchNorm2d(F1)
        )

        # Spatial Convolution (= Depthwise)
        self.DepthwiseConv2D = nn.Sequential(
            nn.Conv2d(F1,
                      F1*D,
                      (n_channels, 1),
                      groups=F1,
                      bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )

        # Block 2: Separable Convolution
        # Separable Convolution
        self.SeparableConv2D = nn.Sequential(
            nn.Conv2d(F1*D,
                      F1*D,
                      (1, 16),
                      padding=(0, 16 // 2),
                      groups=F1*D,
                      bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )

        # Fully connected (classification layer)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F1*D * (samples // (4*8)), n_classes),  # Dense(nb_classes, name='dense')(flatten)
            nn.Softmax(dim=1)
        )

        # output
        out = 0

    def forward(self, x):
        # input x = (channels, samples, 1) / channels=19, samples=time points in EEG data
        x = self.Conv2D(x)  # Block 1
        x = self.DepthwiseConv2D(x)
        x = self.SeparableConv2D(x)  # Block 2
        x = self.classifier(x)  # Flatten and classify
        return x

# model = EEGNet(n_channels=19, n_classes=3, samples=*, dropout_rate=*, sampling_rate=500, F1=*, D=*)
# print(model)
