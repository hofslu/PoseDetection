import bpy
import mathutils as math
import json
import os

def get_camera_parameters(camera):
    # Intrinsic parameters
    focal_length = camera.data.lens  # in millimeters
    sensor_width = camera.data.sensor_width  # in millimeters
    sensor_height = camera.data.sensor_height  # in millimeters
    resolution_x = bpy.context.scene.render.resolution_x  # in pixels
    resolution_y = bpy.context.scene.render.resolution_y  # in pixels
    scale = bpy.context.scene.render.resolution_percentage / 100.0

    # Adjust resolution based on render scale
    resolution_x = int(resolution_x * scale)
    resolution_y = int(resolution_y * scale)

    pixel_aspect_ratio = bpy.context.scene.render.pixel_aspect_x / bpy.context.scene.render.pixel_aspect_y

    # Calculate principal point (assuming center of the image)
    c_x = resolution_x / 2.0
    c_y = resolution_y / 2.0

    # Calculate focal lengths in terms of pixels
    f_in_mm = camera.data.lens
    sensor_size_in_mm = camera.data.sensor_width  # Assuming horizontal sensor fit
    if camera.data.sensor_fit == 'VERTICAL':
        sensor_size_in_mm = camera.data.sensor_height
    elif camera.data.sensor_fit == 'AUTO':
        sensor_fit = 'HORIZONTAL' if resolution_x >= resolution_y else 'VERTICAL'
        if sensor_fit == 'VERTICAL':
            sensor_size_in_mm = camera.data.sensor_height

    focal_length_x = (f_in_mm / sensor_size_in_mm) * resolution_x
    focal_length_y = focal_length_x / pixel_aspect_ratio

    intrinsic_matrix = [
        [focal_length_x, 0, c_x],
        [0, focal_length_y, c_y],
        [0, 0, 1]
    ]

    # Extrinsic parameters
    # Camera rotation and location
    location = camera.matrix_world.to_translation()
    rotation = camera.matrix_world.to_euler('XYZ')

    # Convert rotation to rotation matrix
    rotation_matrix = camera.matrix_world.to_3x3()

    # Prepare the parameters dictionary
    params = {
        'name': camera.name,
        'intrinsic': {
            'focal_length_mm': focal_length,
            'sensor_width_mm': sensor_width,
            'sensor_height_mm': sensor_height,
            'resolution_x_px': resolution_x,
            'resolution_y_px': resolution_y,
            'pixel_aspect_ratio': pixel_aspect_ratio,
            'principal_point_px': [c_x, c_y],
            'focal_length_x_px': focal_length_x,
            'focal_length_y_px': focal_length_y,
            'intrinsic_matrix': intrinsic_matrix
        },
        'extrinsic': {
            'location': [location.x, location.y, location.z],
            'rotation_euler_xyz_deg': [math.degrees(angle) for angle in rotation],
            'rotation_matrix': [list(row) for row in rotation_matrix]
        }
    }

    return params

def main():
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    camera_parameters = []

    for cam in cameras:
        params = get_camera_parameters(cam)
        camera_parameters.append(params)

    # Save parameters to a JSON file
    output_path = os.path.join(bpy.path.abspath("//"), "camera_parameters.json")
    with open(output_path, 'w') as f:
        json.dump(camera_parameters, f, indent=4)

    print(f"Camera parameters have been saved to {output_path}")

if __name__ == "__main__":
    main()
