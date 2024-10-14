import subprocess
import os


class BlenderConnector:
    def __init__(self, blender_path, scene_path):
        self.blender_path = blender_path
        self.scene_path = scene_path
        self.detection_objects = []

    def execute_script(self, script_path, args=None):
        # check valid script path
        if not os.path.exists(script_path):
            raise FileNotFoundError("Script path is not valid.")

        # check script ends with exit()
        # needed for Blender to stop execution
        with open(script_path, "r") as f:
            script = f.read()
            if not script.endswith("exit()"):
                raise ValueError("Script does not end with exit()")

        # execute script
        command = [
            self.blender_path,
            self.scene_path,
            '--python',
            script_path,
            args
        ]
        print(command)
        subprocess.run(command)

        # os.system(f"{self.blender_path} {self.scene_path} --python {script_path} {args}")
        return True
    
    def add_detection_object(self, detection_object):
        self.detection_objects.append(detection_object)
    
    def get_detection_object(self, name):
        for detection_object in self.detection_objects:
            if detection_object.name == name:
                return detection_object
        return None

