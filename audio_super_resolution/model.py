from typing import no_type_check_decorator
import torch
import torch.nn as nn


class SineActivation(nn.Module):
    def __init__(self):
        super(SineActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class Bottleneck(nn.Module):
    def __init__(self, planes, kernel_size):
        super(Bottleneck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(planes, planes // 4, kernel_size=1),
            nn.BatchNorm1d(planes // 4),
            SineActivation(),

            nn.Conv1d(
                planes // 4,
                planes // 4,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
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
    def __init__(self, n_layers=[3, 5, 5, 3], init_planes=32):
        super(ResNet, self).__init__()
        self.n_layers = n_layers
        self.planes = init_planes

        # channel up
        self.conv1 = nn.Conv1d(1, self.planes, kernel_size=21, padding=10)
        # Residual blocks
        self.layer1 = self._make_layer(
            Bottleneck,
            n_layers[0],
            self.planes,
            21,
            'up'
        )
        self.layer2 = self._make_layer(
            Bottleneck,
            n_layers[1],
            self.planes,
            21,
            'up'
        )
        self.layer3 = self._make_layer(
            Bottleneck,
            n_layers[2],
            self.planes,
            21,
            'down'
        )
        self.layer4 = self._make_layer(
            Bottleneck,
            n_layers[3],
            self.planes,
            21,
            'down'
        )
        # channel reduce
        self.conv2 = nn.Conv1d(self.planes, 1, kernel_size=11, padding=5)

    def _make_layer(self, block, n_blocks, planes, kernel_size, expand):
        layers = []
        for i in range(n_blocks):
            layers.append(block(planes, kernel_size))
        if expand == 'up':
            layers.append(nn.Conv1d(planes, 2 * planes, kernel_size=1))
            self.planes *= 2
        elif expand == 'down':
            layers.append(nn.Conv1d(planes, planes // 2, kernel_size=1))
            self.planes //= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv2(x)
        return x


if __name__ == '__main__':
    model = ResNet()
    x = torch.randn((4, 1, 16000))
    out = model(x)
