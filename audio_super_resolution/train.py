import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader import AudioLoader
from model import ResNet

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

ds = AudioLoader(
    '/home/aj/repo/snippets/audio_super_resolution/resources/wav',
    'resources/duration_list_prune.json',
    32000
)
dl = DataLoader(ds, batch_size=8, num_workers=8, pin_memory=False)

model = ResNet(n_layers=[3, 5, 5, 3], init_planes=32)
model.to(device)

log = SummaryWriter('log')

criterion = torch.nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=.001)

counter = 0
for e in range(1000):
    print('-' * 20 + f'epoch: {e+1:02d}' + '-' * 20)
    for x, y in tqdm(dl):
        x = x.to(device)
        y = y.to(device)
        out = model(x)

        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log.add_scalar('loss', loss.item(), counter)
        counter += 1
    torch.save(
        model.state_dict(),
        f'models/asr_e{e+1:02d}.pth'
    )
