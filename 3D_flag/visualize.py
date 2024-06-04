from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from tqdm import tqdm

path = Path("/tmp")
positions = sorted(path.glob("*.pt"))

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


# create faces array, only for z=0
NUM_X_POINTS = 3
NUM_Y_POINTS = 2
NUM_Z_POINTS = 1
faces = []
for j in range(NUM_Y_POINTS - 1):
    for i in range(NUM_X_POINTS - 1):
        faces.append([2, i + j * NUM_X_POINTS, i + 1 + j * NUM_X_POINTS, i + NUM_X_POINTS + j * NUM_X_POINTS])
        faces.append([2, i + 1 + j * NUM_X_POINTS, i + 1 + NUM_X_POINTS + j * NUM_X_POINTS, i + NUM_X_POINTS + j * NUM_X_POINTS])
faces = np.array(faces, dtype=np.int32) + 1
print(faces)
# from IPython import embed
# embed()

for t, position_pt in enumerate(tqdm(positions)):
    plotter.clear()
    position = torch.load(position_pt).view(-1, 3).cpu().numpy()
    # mesh = pv.PolyData(position, faces)
    mesh = pv.PolyData(position, faces=faces)
    mesh = mesh.rotate_x(90)
    plotter.add_mesh(mesh, show_edges=True)

    # Display the rendering scene
    image = plotter.screenshot(f'/tmp/zrender{t:04d}.png')

    print(position)
    if t == 10:
        break

# from IPython import embed
# embed()
# ffmpeg -framerate 30 -i /tmp/zrender%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4
