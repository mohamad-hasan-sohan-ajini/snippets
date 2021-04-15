import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader import AudioLoader
from model import Detector2, ResNet

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
dl = DataLoader(ds, batch_size=5, num_workers=5, pin_memory=True)

model = ResNet(n_layers=[3, 5, 5, 3], init_planes=32)
model.to(device)

log = SummaryWriter('log')

criterion = torch.nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=.1
)

# add gan stuff
detector = Detector2()
detector.to(device)
detector_optimizer = torch.optim.RMSprop(detector.parameters(), lr=.0005)
generator_optimizer = torch.optim.RMSprop(model.parameters(), lr=.0005)
one = torch.FloatTensor([1]).to(device)
mone = torch.FloatTensor([-1]).to(device)

torch.autograd.set_detect_anomaly(True)
counter = 0
for e in range(1000):
    print('-' * 20 + f'epoch: {e+1:02d}' + '-' * 20)
    for x, y in tqdm(dl):
        # prepare input and target
        x = x.to(device)
        y = y.to(device)

        # train updampling model
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log.add_scalar('loss', loss.item(), counter)

        # train detector
        # with real batch
        start = torch.randint(0, 32000 - 33, (1,))
        end = start + 33
        real = y[:, :, start:end]
        detector_error_real = detector(real).mean(dim=0)
        detector_optimizer.zero_grad()
        detector_error_real.backward(one)
        # with fake batch
        fake = out[:, :, start:end].detach()
        detector_error_fake = detector(fake).mean(dim=0)
        detector_error_fake.backward(mone)
        detector_optimizer.step()
        log.add_scalar(
            'detector-error',
            detector_error_real.item() - detector_error_fake.item(),
            counter
        )

        # train generator
        if counter % 5 == 0:
            generator_optimizer.zero_grad()
            out = model(x)
            generator_error = detector(out).mean(dim=0)
            generator_error.backward(one)
            generator_optimizer.step()
            log.add_scalar('generator-error', generator_error.item(), counter)

        # update counter
        counter += 1

    torch.save(model.state_dict(), f'models/asr_e{e+1:02d}.pth')
    scheduler.step()
