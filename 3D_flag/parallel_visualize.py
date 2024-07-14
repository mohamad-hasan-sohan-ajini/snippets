
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from tqdm import tqdm

path = Path("/tmp")
positions = sorted(path.glob("*.pt"))

# create faces array, only for z=0
NUM_X_POINTS = 100
NUM_Y_POINTS = 70
NUM_Z_POINTS = 1
faces = []
for j in range(NUM_Y_POINTS - 1):
    for i in range(NUM_X_POINTS - 1):
        # p2----p1
        # |      |
        # |      |
        # p3----p0
        p0 = j + i * NUM_Y_POINTS
        p1 = p0 + 1
        p2 = p1 + NUM_Y_POINTS
        p3 = p2 - 1
        faces.append([3, p0, p1, p2])
        faces.append([3, p0, p2, p3])
faces = np.array(faces, dtype=np.int32)


def process_frame(position_pt):
    pv.start_xvfb()

    # Create a PyVista plotter
    plotter = pv.Plotter(
        window_size=(1920, 1080),
        # notebook=True,
        off_screen=True,
    )

    # Set up plotter options (if needed)
    plotter.set_background('black')
    plotter.show_axes()

    # set camera
    plotter.camera.position = np.array([.75, 3, .5])
    plotter.camera.focal_point = np.array([.75, 0, .5])
    plotter.camera.view_up = np.array([1, 0, 0])
    plotter.camera_set = True

    plotter.clear()
    position = torch.load(position_pt).view(-1, 3).cpu().numpy()
    mesh = pv.PolyData(position, faces=faces)
    mesh = mesh.rotate_x(90)
    plotter.add_mesh(mesh, show_edges=True)

    # extract frame number
    match = re.search(r'\d+', position_pt.name)
    frame_number = match.group()

    # Display the rendering scene
    image = plotter.screenshot(f'/tmp/zrender{frame_number}.png')
    plotter.close()


with Pool(40) as p:
    for result in tqdm(p.imap_unordered(process_frame, positions), total=len(positions)):
        pass


# from IPython import embed
# embed()
# ffmpeg -framerate 30 -i /tmp/zrender%04d.png -c:v libx264 -pix_fmt yuv420p -y output.mp4
