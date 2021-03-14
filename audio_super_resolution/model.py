from typing import no_type_check_decorator
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
        return self.sin_activation(identity + x)


class ResNet(nn.Module):
    def __init__(self, n_layers=3, n_planes=[32, 64, 128, 64, 32], tied=True):
        super(ResNet, self).__init__()
        self.n_layers = n_layers
        self.n_planes = n_planes
        self.conv1 = nn.Conv1d(1, n_planes[0], kernel_size=11, padding=5)
        self.resnet = nn.Sequential()
        for plane_index, planes in enumerate(n_planes):
            for layer in range(n_layers):
                self.resnet.add_module('Block{plane_index:01d}_layer{layer:01d}', Bottleneck(planes))
        self.conv1_reverse = nn.Conv1d(n_planes[-1], 1, kernel_size=11, padding=5)
        if n_planes[0] == n_planes[-1] and tied:
            self.conv1_reverse.weight.data = self.conv1.weight.transpose(0, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet(x)
        x = self.conv1_reverse(x)
        return x


if __name__ == '__main__':
    model = ResNet()
    x = torch.randn((4, 1, 16000))
    y = model(x)
