import torch
import torchaudio
import torch.nn as nn


class SpectLoss(nn.Module):
    def __init__(self, lowest_bin):
        super(SpectLoss, self).__init__()
        self.lb = lowest_bin
        self.transform = torchaudio.transforms.Spectrogram(
            n_fft=400,
            hop_length=160
        )
        self.criterion = nn.SmoothL1Loss(beta=.3)

    def get_spect(self, x):
        spect = self.transform(x)
        spect = spect[:, :, self.lb:] + 1e-10
        return spect.log10()

    def forward(self, x, x_hat):
        spect = self.get_spect(x)
        spect_hat = self.get_spect(x_hat)
        loss = self.criterion(spect, spect_hat)
        return loss
