import torch
import torch.nn as nn


class SineActivation(nn.Module):
    def __init__(self):
        super(SineActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class Bottleneck(nn.Module):
    def __init__(self, planes):
        super(Bottleneck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(planes, planes // 4, kernel_size=1),
            nn.BatchNorm1d(planes // 4),
            SineActivation(),

            nn.Conv1d(planes // 4, planes // 4, kernel_size=11, padding=5),
            nn.BatchNorm1d(planes // 4),
            SineActivation(),

            nn.Conv1d(planes //4, planes, kernel_size=1),
            nn.BatchNorm1d(planes),
        )
        self.sin_activation = SineActivation()

    def forward(self, x):
        identity = x
        x = self.conv(x)
        return self.sine_activation(identity + x)


class ResNet(nn.Module):
    def __init__(self, n_layers=3, planes=[32, 64, 128, 64, 32]):
        super(ResNet, self).__init__()
        self.n_layers = n_layers
        self.planes = planes

    def forward(self, x):
        pass
