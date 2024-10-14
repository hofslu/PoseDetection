"""
Build out an flask server serving sample routes for testing the frontend behavior
    / - returns a simple information string
    /api - registers the api routes and displays them, each route shows of an info page with the expected handover information on the POST request when served for GET
        /place
            explanation: places the detecion Person in the simulator 
            at given place with given orientation
            assert: {
                "position": list of 3 floats,
                "orientation": list of 4 floats
            }
            return: {
                "command": string - "placed"
            }
        /extract
            explanation: extracts the Camera views from the simulator
            assert: {
                "command": string - "continue"
            }
            return: {
                "extraction_uuid": string - "uuid"
                }
        /detect
            explanation: for each Camera view of extraction_uuid, detect keypoints
            assert: {
                "extraction_uuid": string - "uuid",
            }
            return: {
                "command": "detected"
            }
Host: 0.0.0.0
Port: 5000
"""

from utils.BlenderConnector import BlenderConnector
from utils.DetectionObject import DetectionObject
from utils.utils import validate_position, extract_images

from flask import Flask, request, jsonify
from flask_cors import CORS

import uuid
import subprocess
import time
import os

import numpy as np

origins = [
    [-2.007873296737671, 2.1570141315460205, 2.6291472911834717],
    [2.007873058319092, -2.157014846801758, 2.62914776802063],
    [-2.007873296737671, -2.1570146083831787, 2.6291472911834717],
    [2.007873296737671, 2.1570141315460205, 2.6291472911834717]
]


jack = DetectionObject([0, 0, 0], [0, 0, 0, 1], "Jack")

blenderConnector = BlenderConnector(
    blender_path="/Applications/Blender.app/Contents/MacOS/Blender",
    scene_path="/Users/holu/Documents/tinkering/PoseDetection/simulator/blender_data/simulator.blend"
    )


app = Flask(__name__)
CORS(app)

# Store the last POST request data for each route
last_requests = {
    "place": [],
    "extract": [],
    "detect": [],
    "visualize": []
}

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/api')
def api():
    place_request_list = [f"<li>{r}</li>" for r in last_requests['place']]
    extract_request_list = [f"<li>{r}</li>" for r in last_requests['extract']]
    detect_request_list = [f"<li>{r}</li>" for r in last_requests['detect']]
    visualize_request_list = [f"<li>{r}</li>" for r in last_requests['visualize']]
    # return list of all requests

    if len(place_request_list) >0:
        place_string = f"""<li><a href="/api/place">Place</a> - {last_requests['place'][-1]}</li>"""
    else:
        place_string = """<li><a href="/api/place">Place</a></li>"""

    if len(extract_request_list) > 0:
        extract_string = f"""<li><a href="/api/extract">Extract</a> - {last_requests['extract'][-1]}</li>"""
    else:
        extract_string = """<li><a href="/api/extract">Extract</a></li>"""

    if len(detect_request_list) >0:
        detect_string = f"""<li><a href="/api/detect">Detect</a> - {last_requests['detect'][-1]}</li>"""
    else:
        detect_string = """<li><a href="/api/detect">Detect</a></li>"""

    if len(visualize_request_list) > 0:
        visualize_string = f"""<li><a href="/api/visualize">Visualize</a> - {last_requests['visualize'][-1]}</li>"""
    else:
        visualize_string = """<li><a href="/api/visualize">Visualize</a></li>"""

    return f"""
    <h1>API</h1>
    <ul>
        {place_string}
        {extract_string}
        {detect_string}
        {visualize_string}
    </ul>
    <h4>place</h4>
    <ul>
        {place_request_list}
    </ul>
    <h4>extract</h4>
    <ul>
        {extract_request_list}
    </ul>
    <h4>detect</h4>
    <ul>
        {detect_request_list}
    </ul>
    <h4>visualize</h4>
    <ul>
        {visualize_request_list}
    </ul>
    """

@app.route('/api/place', methods=['GET', 'POST', 'OPTIONS'])
def place():
    if request.method == 'POST':
        if request.content_type == 'application/json':
            data = request.get_json()
        else:
            data = request.form
        
        if data is None:
            return jsonify({"error": "Invalid JSON payload"}), 400

        position = data.get("position")
        orientation = data.get("orientation")

        if not position or not orientation:
            return jsonify({"error": "Missing key position or orientation"}), 400
        
        # Validate the position and orientation
        error = validate_position(position)
        if error:
            return jsonify({"error": str(error)}), 400
        
        last_requests["place"].append({"position": position, "orientation": orientation})
        

        position[1] = 0 # [TESTING] Set y position to 0
        position = [p*4 for p in position] # DEBUGGING
        print("Position:", position)
        print("Orientation:", orientation)

        jack.set_position(position)
        jack.set_orientation(orientation)

        return jsonify({"command": "placed"})
    
    # Handle OPTIONS request for preflight CORS
    if request.method == 'OPTIONS':
        return jsonify({"status": "CORS preflight check OK"}), 200

    return f"""
    <h1>Place</h1>
    <p>Last POST requests:</p>
    <pre>{last_requests["place"]}</pre>
    <form method="post">
        <input type="text" name="position" placeholder="position">
        <input type="text" name="orientation" placeholder="orientation">
        <button type="submit">Place</button>
    </form>
    """

@app.route('/api/extract', methods=['GET', 'POST'])
def extract():
    if request.method == 'POST':
        if request.content_type == 'application/json':
            data = request.get_json()
        else:
            data = request.form

        command = data.get("command")
        print("[EXTRACTION] Command:", command)

        if command != "extract":
            return jsonify({"error": "Invalid command"}), 400
        time.sleep(2)
        extraction_uuid = str(uuid.uuid4())
        last_requests["extract"].append({"command": command})

        # place jack in simulation and extrat camera views
        # DONE: camera view extraction
        # TODO: pass jack position an rotation to blender_script
        blenderConnector.execute_script(
            script_path="/Users/holu/Documents/tinkering/PoseDetection/simulator/capture_camera_view.py",
            args=f"--position {jack.position} --orientation {jack.orientation}"
        )
        return jsonify({"extraction_uuid": extraction_uuid})

    return f"""
    <h1>Extract</h1>
    <p>Last POST requests:</p>
    <pre>{last_requests["extract"]}</pre>
    <form method="post">
        <input type="text" name="command" placeholder="command">
        <button type="submit">Extract</button>
    </form>
    """

@app.route('/api/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if request.content_type == 'application/json':
            data = request.get_json()
        else:
            data = request.form

        extraction_uuid = data.get("extraction_uuid")

        print("[DETECTION] Extraction UUID:", extraction_uuid)
        if not extraction_uuid:
            return jsonify({"error": "Missing extraction_uuid"}), 400

        last_requests["detect"].append({"extraction_uuid": extraction_uuid})

        # TODO: check sent uuid against local_extraction_uuid
        # TODO: upload camera views to detection service (maybe we 
        #   let the extraction process run on the same machine as 
        #   the detecion service)
        # TODO: start detection process
        # TODO: return detection_uuid

        return jsonify({"command": "detected"})

    return f"""
    <h1>Detect</h1>
    <p>Last POST requests:</p>
    <pre>{last_requests["detect"]}</pre>
    <form method="post">
        <input type="text" name="extraction_uuid" placeholder="extraction_uuid">
        <button type="submit">Detect</button>
    </form>
    """

@app.route('/api/visualize', methods=['GET', 'POST'])
def visualize():
    if request.method == 'POST':
        if request.content_type == 'application/json':
            data = request.get_json()
        else:
            data = request.form

        extraction_uuid = data.get("extraction_uuid")

        print("[VISUALIZE] Extraction UUID:", extraction_uuid)
        if not extraction_uuid:
            return jsonify({"error": "Missing extraction_uuid"}), 400

        last_requests["visualize"].append({"extraction_uuid": extraction_uuid})

        # TODO: get detection points on each camera view
        # TODO: build rays for each view
        # TODO: triangulate the detected points
        # TODO: build points and rays return-list

        detectionPoints = np.random.rand(16, 3).tolist()
        # build rays in form each origin to each detection point
        # origin: holds the x,y,Z of the ray origin
        # direction: holds the x,y,z of the ray direction towards the detection point
        detectionRays = []
        for origin in origins:
            for detectionPoint in detectionPoints:
                detectionRays.append({
                    "origin": origin,
                    "direction": detectionPoint
                })

        return jsonify({
            "command": "visualized",
            "detectionPoints": detectionPoints,
            "detectionRays": detectionRays
            })

    return f"""
    <h1>Visualize</h1>
    <p>Last POST requests:</p>
    <pre>{last_requests["visualize"]}</pre>
    <form method="post">
        <input type="text" name="extraction_uuid" placeholder="extraction_uuid">
        <button type="submit">Visualize</button>
    </form>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
