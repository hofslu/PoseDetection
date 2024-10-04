import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import json 

points_and_rays_path = 'experiments/points_and_rays.json'

with open(points_and_rays_path, 'r') as f:
    points_and_rays = json.load(f)

"""
example point: [-2.007873296737671, 2.1570141315460205, 2.6291472911834717]
example ray: {'origin': [2.007873058319092, -2.157014846801758, 2.62914776802063], 'direction': [-0.5647206844325035, 0.5647205050436787, -0.6018150045963438]}
"""
print(points_and_rays)


# Initialize the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for point in points_and_rays['points']:
    ax.scatter(point[0], point[1], point[2], color='red')
for ray in points_and_rays['rays']:
    origin = ray['origin']
    direction = ray['direction']
    ax.quiver(origin[0], origin[1], origin[2], direction[0], direction[1], direction[2], color='blue')
    length = 5
    end_point = np.array(origin) + np.array(direction) * length
    ax.plot([origin[0], end_point[0]], [origin[1], end_point[1]], [origin[2], end_point[2],], color='blue', linestyle='--')
ax.set_xlabel('X')
ax.set_xlim(-5, 5)
ax.set_ylabel('Y')
ax.set_ylim(-5, 5)
ax.set_zlabel('Z')
ax.set_zlim(0, 10)
ax.set_title('Points and Rays')
# enable zooming
ax.view_init(elev=20, azim=30)
# Equal aspect ratio for all axes
ax.set_box_aspect([1,1,1])
plt.show()
