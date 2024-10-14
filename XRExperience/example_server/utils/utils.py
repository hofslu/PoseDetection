import os

def validate_position(position):
    if not isinstance(position, list):
        return TypeError("Position must be a list")
    if not len(position) == 3:
        return ValueError("Position must have 3 elements")
    return None


def extract_images(position=None):
    return DeprecationWarning("moved to BlenderConnector")



