import json
import os

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class AudioLoader(Dataset):
    def __init__(self, base_path, json_file_list, n_samples):
        self.base_path = base_path
        with open(json_file_list) as f:
            self.data = json.load(f)
        self.n_samples = n_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_name = self.data[index][0]
        file_path = os.path.join(self.base_path, file_name)
        y, sr = sf.read(file_path)

        # TODO: unit gain
        # crop signal
        start = np.random.randint(0, y.shape[0] - self.n_samples)
        end = start + self.n_samples
        y = y[start:end].astype(np.float32)

        # create masked input
        x = np.copy(y)
        x[np.random.randint(0, 2)::2] = 0

        # TODO: bulk masking
        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.from_numpy(y).unsqueeze(0)
        return x, y


if __name__ == '__main__':
    al = AudioLoader(
        '/home/aj/repo/snippets/audio_super_resolution/resources/wav',
        'resources/duration_list_prune.json',
        32000
    )
    for x, y in tqdm(al):
        break
