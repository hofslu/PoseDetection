


class DetectionObject:
    def __init__(self, position, orientation, name=None):
        self.position = position
        self.orientation = orientation
        self.name = "name"
    
    def set_position(self, position):
        self.position = position
    
    def set_orientation(self, orientation):
        self.orientation = orientation
    
    def set_extraction_uuid(self, extraction_uuid):
        self.extraction_uuid = extraction_uuid
    
    def get_extraction_uuid(self):
        if not hasattr(self, "extraction_uuid"):
            return None
        return self.extraction_uuid
    
    def check_valid_continue(self, extraction_uuid):
        if not hasattr(self, "extraction_uuid"):
            return KeyError("No extraction_uuid set")
        if extraction_uuid != self.extraction_uuid:
            return ValueError("Extraction UUID does not match")
        return True