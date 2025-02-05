import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from tqdm import tqdm

pv.start_xvfb()

# List of .obj file paths
base_path = Path('results')
obj_files = sorted(base_path.rglob('*.obj'))
obj_files = obj_files[:20_000]

# Create a PyVista plotter
plotter = pv.Plotter(
    window_size=(1920, 1080),
    # notebook=True,
    off_screen=True,
)

# Set up plotter options (if needed)
plotter.set_background('black')
# plotter.show_axes()

# Load and add each object to the plotter
print('Loaindg meshes...')
green_texture = pv.Texture('Tex/GrassG.png')
white_texture = pv.Texture('Tex/GrassW.png')
max_x, sup_y, max_z = 0, 0, 0
for obj_file in tqdm(obj_files, file=sys.stdout):
    mesh = pv.read(obj_file)
    max_x = max(mesh.bounds[1], max_x)
    sup_y = min(mesh.bounds[3], sup_y)
    max_z = max(mesh.bounds[5], max_z)
    if '-G' in obj_file.name:
        plotter.add_mesh(mesh, texture=green_texture)
    else:
        plotter.add_mesh(mesh, texture=white_texture)

# moving camera settings
timesteps = 300
start_position = np.array([max_x, sup_y, max_z / 2])
mid_position = np.array([max_x / 2, 1, max_z / 2])
end_position = np.array([max_x / 2, 5, max_z / 2])

start_look_at = np.array([max_x, 0.0, max_z / 2])
mid_look_at = np.array([0.0, 0.0, max_z / 2])
end_look_at = np.array([max_x / 2, 0.0, max_z / 2])

# linspace
position_linspace = np.concatenate([np.linspace(start_position, mid_position, timesteps // 2), np.linspace(mid_position, end_position, timesteps // 2)])
look_at_linspace = np.concatenate([np.linspace(start_look_at, mid_look_at, timesteps // 2), np.linspace(mid_look_at, end_look_at, timesteps // 2)])

# MA filter
kernel_size = (timesteps // 10) + 1
position_linspace0 = np.convolve(position_linspace[:, 0], np.ones(kernel_size), 'valid') / kernel_size
position_linspace1 = np.convolve(position_linspace[:, 1], np.ones(kernel_size), 'valid') / kernel_size
position_linspace2 = np.convolve(position_linspace[:, 2], np.ones(kernel_size), 'valid') / kernel_size
position_linspace = np.stack([position_linspace0, position_linspace1, position_linspace2], axis=1)

look_at_linspace0 = np.convolve(look_at_linspace[:, 0], np.ones(kernel_size), 'valid') / kernel_size
look_at_linspace1 = np.convolve(look_at_linspace[:, 1], np.ones(kernel_size), 'valid') / kernel_size
look_at_linspace2 = np.convolve(look_at_linspace[:, 2], np.ones(kernel_size), 'valid') / kernel_size
look_at_linspace = np.stack([look_at_linspace0, look_at_linspace1, look_at_linspace2], axis=1)

print('Rendering...')
for t, (position, look_at) in enumerate(zip(position_linspace, tqdm(look_at_linspace, file=sys.stdout))):
    # set up camera
    plotter.camera.position = position
    plotter.camera.focal_point = look_at
    plotter.camera.view_up = np.array([-1, 0, 0])
    plotter.camera_set = True

    # Display the rendering scene
    plotter.show(interactive=False, auto_close=False)
    image = plotter.screenshot(f'/tmp/zrender{t:04d}.png')

# ffmpeg -framerate 30 -i frame%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4
