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
        self.criterion = nn.MSELoss()

    def get_spect(self, x):
        spect = self.transform(x)
        return spect[:, :, self.lb:]

    def forward(self, x, x_hat):
        spect = self.get_spect(x)
        spect_hat = self.get_spect(x_hat)
        loss = self.criterion(spect, spect_hat)
        return loss
