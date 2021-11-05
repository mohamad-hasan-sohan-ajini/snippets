import asyncio
import base64
import multiprocessing
import queue

from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # q = queue.Queue()
    q = multiprocessing.Queue()
    # while True:
    #     try:
    #         b64data = await asyncio.wait_for(websocket.receive_bytes(), 10.)
    #         data = base64.b64decode(b64data)
    #         await websocket.send_text(
    #             f'b64 len: {len(b64data)}, raw len: {len(data)}'
    #         )
    #     except asyncio.TimeoutError:
    #         break

    async def consumer(websocket, q):
        while True:
            try:
                b64data = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    10.,
                )
                data = base64.b64decode(b64data)
                q.put_nowait(data)
                print(
                    f'@consumer: b64 len: {len(b64data)},'
                    ' bytes len: {len(data)}'
                )
            except asyncio.TimeoutError:
                print('@consumer: timeout exception')
                break

    async def producer(websocket, q):
        counter = 0
        while True:
            counter += 1
            try:
                # TODO: read from queue
                if not q.empty():
                    data = q.get_nowait()
                    print('@sender: processing some data...')
                    await asyncio.wait_for(
                        websocket.send_text('server processed some data'),
                        10.,
                    )
                    counter = 0
                await asyncio.sleep(.1)
            except asyncio.TimeoutError:
                pass

    loop = asyncio.get_event_loop()
    consumer_task = loop.create_task(consumer(websocket, q))
    producer_task = loop.create_task(producer(websocket, q))
    await asyncio.wait([consumer_task, producer_task])
