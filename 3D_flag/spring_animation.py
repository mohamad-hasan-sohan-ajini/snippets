import numpy as np
import matplotlib.pyplot as plt
import time

# Constants
obj1_mass = 1
obj1_position = np.array([-5.0, 0.0])
obj1_velocity = np.array([0.0, 0.0])
obj2_mass = 1
obj2_position = np.array([5.0, 0.0])
obj2_velocity = np.array([0.0, 0.0])

ks = 0.1
kd = 0.01
rest_length = 6
delta_t = 0.1

# Function to compute the spring force
def spring_force(pos1, pos2, vel1, vel2, ks, kd, rest_length):
    distance_vector = pos2 - pos1
    distance = np.linalg.norm(distance_vector)
    direction = distance_vector / distance if distance != 0 else np.zeros(2)

    force_magnitude = ks * (distance - rest_length)
    damping_force_magnitude = kd * np.dot(vel2 - vel1, direction)

    force = (force_magnitude + damping_force_magnitude) * direction
    return force

# Initialize the plot
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
line, = ax.plot([obj1_position[0], obj2_position[0]], [obj1_position[1], obj2_position[1]], 'bo-', lw=2)

# Simulation loop
for _ in range(200):  # Adjust the range for a longer or shorter simulation
    # Compute the spring force
    force = spring_force(obj1_position, obj2_position, obj1_velocity, obj2_velocity, ks, kd, rest_length)

    # Update velocities
    obj1_velocity += (force / obj1_mass) * delta_t
    obj2_velocity -= (force / obj2_mass) * delta_t

    # Update positions
    obj1_position += obj1_velocity * delta_t
    obj2_position += obj2_velocity * delta_t

    # Update the line data
    line.set_xdata([obj1_position[0], obj2_position[0]])
    line.set_ydata([obj1_position[1], obj2_position[1]])

    # Redraw the plot
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(delta_t)

# Keep the final plot visible
plt.ioff()
plt.show()
