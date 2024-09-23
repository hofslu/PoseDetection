import bpy
from datetime import datetime
import time
import os

current_time = datetime.now().strftime("%Y%m%d%H%M%S")

# Set the output directory
output_dir = f'/Users/holu/Documents/tinkering/PoseDetection/simulator/data/{current_time}/'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set render settings
scene = bpy.context.scene
scene.render.image_settings.file_format = 'PNG'

# Get the 'Cameras' collection
cameras_collection = bpy.data.collections.get('Cameras')
if cameras_collection:
    cameras = [obj for obj in cameras_collection.objects if obj.type == 'CAMERA']
else:
    cameras = []

# Get the 3D View area and space
for area in bpy.context.window.screen.areas:
    if area.type == 'VIEW_3D':
        space = area.spaces.active
        break

# Loop over each camera and render
for idx, camera in enumerate(cameras):
    # Set the active camera
    scene.camera = camera

    # Update the viewport to use the new camera
    space.camera = camera
    space.region_3d.view_perspective = 'CAMERA'

    # Update the output file path
    output_path = os.path.join(output_dir, f'capture_{idx}.png')
    scene.render.filepath = output_path

    # Perform a viewport render (OpenGL render)
    bpy.ops.render.opengl(write_still=True)
