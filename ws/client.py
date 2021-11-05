import asyncio
import base64
import time

import numpy as np
import torchaudio
import websockets

EPS_TIME = .1
TIMESTEP = 1


async def client():
    async def sender(ws):
        x, fs = torchaudio.load('George_cheat.wav')
        x = (x[0] * 2 ** 16).numpy().astype(np.int16)
        print(f'@sender: x shape: {x.shape}')
        counter = 0
        t0 = time.time()
        while counter < x.shape[0]:
            print(f'@sender: counter: {counter}')
            try:
                data = x[counter:counter + int(TIMESTEP * fs)]
                b64data = base64.b64encode(data.tobytes())
                counter += int(TIMESTEP * fs)
                await asyncio.wait_for(ws.send(b64data), 10.)
                while (time.time() - t0) * fs < counter:
                    await asyncio.sleep(EPS_TIME)
            except Exception as E:
                print(f'@sender: excption: {E}')
                print('@sender: asyncio timeout exception!')
                break
        print('@sender: end of sender')

    async def receiver(ws):
        while True:
            try:
                response = await asyncio.wait_for(ws.recv(), 10.)
                print(f'@receiver: {response}')
            except asyncio.TimeoutError:
                print('@receiver: asyncio timeout exception!')
                break
        await ws.close()
        print('@receiver: end of receiver')

    # run the tasks
    loop = asyncio.get_event_loop()
    async with websockets.connect('ws://localhost:8000/ws') as ws:
        sender_agent = loop.create_task(sender(ws))
        receiver_agent = loop.create_task(receiver(ws))
        await asyncio.wait([sender_agent, receiver_agent])


loop = asyncio.get_event_loop()
loop.run_until_complete(client())
