import torch
import torch.nn as nn
import torchaudio


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
    def __init__(self, n_layers=[3, 4, 4, 3], init_planes=32):
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
            'increase'
        )
        self.layer2 = self._make_layer(
            Bottleneck,
            n_layers[1],
            self.planes,
            21,
            'increase'
        )
        self.layer3 = self._make_layer(
            Bottleneck,
            n_layers[2],
            self.planes,
            21,
            'decrease'
        )
        self.layer4 = self._make_layer(
            Bottleneck,
            n_layers[3],
            self.planes,
            21,
            'decrease'
        )
        # channel reduce
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.planes, 1, kernel_size=5, padding=2),
            SineActivation()
        )

    def _make_layer(self, block, n_blocks, planes, kernel_size, expand):
        layers = []
        for _ in range(n_blocks):
            layers.append(block(planes, kernel_size))
        if expand == 'increase':
            layers.append(nn.Conv1d(planes, 2 * planes, kernel_size=1))
            self.planes *= 2
        elif expand == 'decrease':
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


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(out_ch),
            SineActivation()
        )

    def forward(self, x):
        return self.main(x)


class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(ConvBlock2d, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_ch),
            SineActivation()
        )

    def forward(self, x):
        return self.main(x)


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()

        self.conv = nn.Sequential(
            ConvBlock(1, 16, 11, 1),
            ConvBlock(16, 32, 11, 1),
            ConvBlock(32, 64, 11, 1),
            ConvBlock(64, 128, 11, 1),
        )
        self.linear = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=2)
        x = self.linear(x)
        return x


class Detector2(nn.Module):
    def __init__(self):
        super(Detector2, self).__init__()

        self.conv = nn.Sequential(
            ConvBlock(1, 16, 11, 3),
            ConvBlock(16, 16, 11, 3),
            ConvBlock(16, 32, 11, 3),
            ConvBlock(32, 32, 11, 3),
            ConvBlock(32, 64, 11, 3),
            ConvBlock(64, 64, 11, 3),
            ConvBlock(64, 128, 11, 1),
        )
        self.linear = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=2)
        x = self.linear(x)
        return x


class Detector3(nn.Module):
    def __init__(self):
        super(Detector3, self).__init__()

        self.conv = nn.Sequential(
            torchaudio.transforms.Spectrogram(n_fft=400, hop_length=160),
            ConvBlock2d(1, 16, kernel_size=(11, 5), stride=(3, 2)),
            ConvBlock2d(16, 32, kernel_size=(11, 5), stride=(3, 2)),
            ConvBlock2d(32, 64, kernel_size=(11, 5), stride=(3, 2)),
            ConvBlock2d(64, 128, kernel_size=(3, 5), stride=(1, 2)),
        )
        self.linear = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=3).squeeze()
        x = self.linear(x)
        return x


if __name__ == '__main__':
    model = ResNet()
    x = torch.randn((4, 1, 16000))
    out = model(x)
