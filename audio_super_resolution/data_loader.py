import json
import os

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class AudioLoader(Dataset):
    def __init__(
            self,
            base_path,
            json_file_list,
            n_samples,
            n_mask,
            noise_file=None
    ):
        self.base_path = base_path
        with open(json_file_list) as f:
            self.data = json.load(f)
        self.n_samples = n_samples
        self.n_mask = n_mask
        if noise_file:
            self.noise, _ = sf.read(noise_file)
        else:
            self.noise = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_name = self.data[index][0]
        file_path = os.path.join(self.base_path, file_name)
        y, sr = sf.read(file_path)

        # crop signal
        start = np.random.randint(0, y.shape[0] - self.n_samples)
        end = start + self.n_samples
        y = y[start:end].astype(np.float32)

        # create low resolution input
        x = np.copy(y)
        x[np.random.randint(0, 2)::2] = 0

        # time masking
        if np.random.random() < .5:
            x = self._time_mask(x)

        # freq filter
        if np.random.random() < .5:
            x = self._freq_mask(x, sr)

        # additive noise
        if np.random.random() < .5:
            x = self._add_noise(x)

        x = np.ascontiguousarray(x, dtype=np.float32)
        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.from_numpy(y).unsqueeze(0)
        return x, y

    def _time_mask(self, x):
        start = np.random.randint(0, self.n_samples - self.n_mask)
        end = start + self.n_mask
        x[start:end] = 0
        return x

    def _freq_mask(self, x, sr):
        low = np.random.randint(40, 100)
        high = np.random.randint(3500, 4000)
        sos = butter(10, [low, high], 'bp', fs=sr, output='sos')
        x = sosfiltfilt(sos, x)
        return x

    def _add_noise(self, x):
        if self.noise is None:
            return x
        noise_len = self.noise.shape[0]
        start = np.random.randint(0, noise_len - self.n_samples)
        end = start + self.n_samples
        noise = self.noise[start:end]
        # tune SNR
        noise_snr = np.random.uniform(5, 20)
        x_db = 10 * np.log10(np.mean(x ** 2) + 1e-4)
        noise_db = 10 * np.log10(np.mean(noise ** 2) + 1e-4)
        noise *= np.sqrt(10 ** ((x_db - noise_db - noise_snr) / 10))
        x += noise
        return x


if __name__ == '__main__':
    al = AudioLoader(
        '/home/aj/repo/snippets/audio_super_resolution/resources/wav',
        'resources/duration_list_prune.json',
        32000,
        80,
        'resources/noise/whitenoisegaussian.wav'
    )
    for x, y in tqdm(al):
        break
