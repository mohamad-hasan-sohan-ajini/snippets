import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader import AudioLoader
from model import ResNet
from spect_loss import SpectLoss

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

ds = AudioLoader(
    base_path='/home/aj/repo/snippets/audio_super_resolution/resources/wav',
    json_file_list='resources/duration_list_prune.json',
    n_samples=32000,
    n_mask=80,
    noise_file='resources/noise/whitenoisegaussian.wav'
)
dl = DataLoader(ds, batch_size=8, num_workers=8, pin_memory=True)

model = ResNet(n_layers=[3, 5, 5, 3], init_planes=32)
model.to(device)

log = SummaryWriter('log_spect')

criterion1 = torch.nn.SmoothL1Loss(beta=.01)
criterion2 = SpectLoss(lowest_bin=1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=.001)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=.5
)

counter = 0
for e in range(1000):
    print('-' * 20 + f'epoch: {e+1:02d}' + '-' * 20)
    for x, y in tqdm(dl):
        x = x.to(device)
        y = y.to(device)
        out = model(x)

        loss1 = criterion1(out, y)
        loss2 = criterion2(out, y)
        loss = .95 * loss1 + .05 * loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log.add_scalar('loss1', loss1.item(), counter)
        log.add_scalar('loss2', loss2.item(), counter)
        log.add_scalar('loss', loss.item(), counter)
        counter += 1
    torch.save(
        model.state_dict(),
        f'models/asr_e{e+1:02d}.pth'
    )
    scheduler.step()
