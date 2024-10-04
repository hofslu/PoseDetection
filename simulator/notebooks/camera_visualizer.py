
import json
import numpy as np


class Camera:
    def __init__(self, name, intrinsic_params, extrinsic_params):
        self.name = name
        
        # Intrinsic Parameters
        self.focal_length_mm = intrinsic_params['focal_length_mm']
        self.sensor_width_mm = intrinsic_params['sensor_width_mm']
        self.sensor_height_mm = intrinsic_params['sensor_height_mm']
        self.resolution_x_px = intrinsic_params['resolution_x_px']
        self.resolution_y_px = intrinsic_params['resolution_y_px']
        self.pixel_aspect_ratio = intrinsic_params['pixel_aspect_ratio']
        self.principal_point_px = intrinsic_params['principal_point_px']
        self.focal_length_x_px = intrinsic_params['focal_length_x_px']
        self.focal_length_y_px = intrinsic_params['focal_length_y_px']
        self.K = np.array(intrinsic_params['intrinsic_matrix'])
        
        # Extrinsic Parameters
        self.location = np.array(extrinsic_params['location'])  # Translation vector
        self.rotation_euler_deg = np.array(extrinsic_params['rotation_euler_xyz_deg'])
        self.rotation_matrix = np.array(extrinsic_params['rotation_matrix'])
        
        # Convert rotation matrix to 3x3 numpy array if it's a list of lists
        if self.rotation_matrix.shape != (3, 3):
            self.rotation_matrix = np.array(self.rotation_matrix).reshape(3, 3)
        
        # Calculate the camera projection matrix
        self.P = self.calculate_projection_matrix()
    
    def calculate_projection_matrix(self):
        # Transpose of R to get rotation from world to camera coordinates
        R_world_to_camera = self.rotation_matrix.T
        # Compute translation vector t = -R * C
        t = -R_world_to_camera @ self.location.reshape(3, 1)
        # Form the extrinsic matrix [R | t]
        RT = np.hstack((R_world_to_camera, t))
        # Compute the projection matrix
        P = self.K @ RT
        return P


print()
# load in camera parameters
print("Loading camera parameters...")
camera_parameters_example_path = 'experiments/camera-example-parameters/camera_parameters_20241002131616.json'
with open(camera_parameters_example_path, 'r') as f:
    cameras_parameters = json.load(f)
print(cameras_parameters)

print()
# load in experiment data
print("Loading experiment data...")
experiment_data_path = 'experiments/single-person-detection/detection_data.json'
with open(experiment_data_path, 'r') as f:
    experiment_data = json.load(f)
process_response_data_cleaned = experiment_data['data']
print(process_response_data_cleaned)


cameras = []
for cam_params in cameras_parameters:
    name = cam_params['name']
    intrinsic = cam_params['intrinsic']
    extrinsic = cam_params['extrinsic']
    
    # Create a Camera instance
    camera = Camera(name, intrinsic, extrinsic)
    cameras.append(camera)
    
    # Optional: Print camera details to verify
    print(f"Initialized Camera: {camera.name}")
    print(f"Projection Matrix P:\n{camera.P}\n")



def plot_camera(ax, camera, scale=0.5, color='blue'):
    """
    Plots a camera in 3D space.

    Parameters:
    - ax: Matplotlib 3D axis
    - camera: Camera object
    - scale: Scale for the camera size
    - color: Color of the camera frustum
    """
    # Camera center
    C = camera.location

    # Camera orientation (rotation matrix)
    R = camera.rotation_matrix

    # Define the image plane in camera coordinate system
    # Let's define a rectangle at Z = 1 (arbitrary distance)
    w = camera.resolution_x_px
    h = camera.resolution_y_px

    # Normalize width and height based on scale
    w_scaled = scale * w / max(w, h)
    h_scaled = scale * h / max(w, h)

    # Image plane corners in camera coordinates
    image_plane_corners = np.array([
        [-w_scaled / 2, -h_scaled / 2, -1],  # Bottom-left
        [w_scaled / 2, -h_scaled / 2, -1],   # Bottom-right
        [w_scaled / 2, h_scaled / 2, -1],    # Top-right
        [-w_scaled / 2, h_scaled / 2, -1],   # Top-left
    ]).T  # Shape: (3, 4)

    # Transform image plane corners to world coordinates
    image_plane_world = (R @ image_plane_corners) + C.reshape(3, 1)

    # Draw the camera center
    ax.scatter(C[0], C[1], C[2], color=color, marker='o')

    # Draw lines from camera center to image plane corners (frustum edges)
    for i in range(4):
        corner = image_plane_world[:, i]
        ax.plot([C[0], corner[0]], [C[1], corner[1]], [C[2], corner[2]], color=color)

    # Draw the image plane as a rectangle
    # Close the loop by repeating the first corner at the end
    corners = np.hstack((image_plane_world, image_plane_world[:, [0]]))
    ax.plot(corners[0, :], corners[1, :], corners[2, :], color=color)


def backproject_keypoints(camera, keypoints):
    rays = []
    K_inv = np.linalg.inv(camera.K)
    R = camera.rotation_matrix
    C = camera.location.reshape(3, 1)
    
    for kp in keypoints:
        u, v = kp
        # Create homogeneous image coordinate
        uv1 = np.array([u, v, 1]).reshape(3, 1)
        
        # Back-project to normalized camera coordinates
        xyz_camera = K_inv @ uv1
        
        # Transform direction to world coordinates
        direction_world = -R @ xyz_camera
        direction_world = direction_world / np.linalg.norm(direction_world)
        
        # Ray starts at camera center, extends in the direction
        ray = {'origin': C.flatten(), 'direction': direction_world.flatten()}
        rays.append(ray)
    return rays


def build_artificial_keypoints(cameras):
    # build artificial keypoints
    # Define image dimensions (from camera intrinsic parameters)
    image_width = cameras[0].resolution_x_px
    image_height = cameras[0].resolution_y_px

    # Define keypoints in pixel coordinates
    # Center of the image
    center_keypoint = (image_width / 2, image_height / 2)

    # Corners of the image
    top_left = (0, 0)
    top_right = (image_width, 0)
    bottom_left = (0, image_height)
    bottom_right = (image_width, image_height)

    # Collect keypoints
    keypoints = []
    keypoints.extend([center_keypoint])
    # keypoints.extend([top_left, top_right, bottom_left, bottom_right])
    return keypoints

artificial_keypoints = build_artificial_keypoints(cameras) 

keypoints_per_camera = []
# List of camera image filenames
camera_image_filenames = [
    'Camera.000.png',
    'Camera.001.png',
    'Camera.002.png',
    'Camera.003.png'
]
for filename in camera_image_filenames:
    keypoints = process_response_data_cleaned[filename]['keypoints']
    # Get the 0th keypoint
    if True:
        kp0 = keypoints[0]
        keypoints_per_camera.append([kp0])  # Wrap in a list for compatibility
    # Use all keypoints
    else:
        keypoints_per_camera.append(keypoints)  # Each element is a list of keypoints for a camera


# Colors for different cameras
camera_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Initialize the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# For each camera, plot the camera and the rays
for idx, camera in enumerate(cameras):
    color = camera_colors[idx % len(camera_colors)]
    plot_camera(ax, camera, scale=0.5, color=color)
    
    actual_keypoints = keypoints_per_camera[idx]

    # Back-project keypoints to obtain rays
    debug_rays = backproject_keypoints(camera, artificial_keypoints)
    actual_rays = backproject_keypoints(camera, actual_keypoints)
    
    rays = []
    rays.extend(debug_rays)
    # rays.extend(actual_rays)

    # Plot rays
    for ray in rays:
        origin = ray['origin']
        direction = ray['direction']
        # Define the length of the ray for visualization
        length = 5  # Adjust as needed
        end_point = origin + direction * length
        ax.plot([origin[0], end_point[0]],
                [origin[1], end_point[1]],
                [origin[2], end_point[2]],
                color=color, linestyle='--')

# Set plot labels
ax.set_xlabel('X')
ax.set_xlim(-5, 5)
ax.set_ylabel('Y')
ax.set_ylim(-5, 5)
ax.set_zlabel('Z')
ax.set_zlim(0, 10)
ax.set_title('Camera Positions and Orientations')

# Equal aspect ratio for all axes
ax.set_box_aspect([1,1,1])

# Show the plot
plt.show()
