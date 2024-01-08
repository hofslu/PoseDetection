from flask import Flask, Response, render_template, jsonify
import cv2
import mediapipe as mp
from threading import Thread
import time

CALIBRATION_TIME = 10

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

import requests

def say(text):
    try:
        requests.post('http://10.1.1.242:5000/speak', json={"say": text})
    except requests.exceptions.RequestException as e:
        print(f"Error sending speech request: {e}")

def make_announcements(camera_threads):
    # Announce the countdown
    time.sleep(CALIBRATION_TIME - 5)
    say("5 seconds left.")
    time.sleep(2)
    say("3")
    time.sleep(1)
    say("2")
    time.sleep(1)
    say("1")
    time.sleep(1)

    # Wait for camera threads to finish
    for thread in camera_threads:
        thread.join()
    say("Calibration capturing finished.")


def capture_frames(camera_url, output_file):
    cap = cv2.VideoCapture(camera_url)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera stream.")
        return

    # Retrieve frame size from the camera stream
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Capturing frames at {}x{}".format(frame_width, frame_height))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))

    start_time = time.time()
    while time.time() - start_time < CALIBRATION_TIME:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break

    cap.release()
    out.release()

@app.route('/capture_for_calibration', methods=['GET'])
def capture_for_calibration():
    say("Starting calibration.")

    # Start capturing frames in separate threads
    thread1 = Thread(target=capture_frames, args=('http://10.1.1.242:5001/stream', 'camera1.avi'))
    thread2 = Thread(target=capture_frames, args=('http://10.1.1.211:4747/video', 'camera2.avi'))
    thread1.start()
    thread2.start()

    # Start the announcements in a separate thread and pass camera threads
    Thread(target=make_announcements, args=([thread1, thread2],)).start()

    return jsonify({"status": "Capturing frames for calibration"})


def mediapipe_process(frame, pose):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return frame

def gen_frames(camera_url):
    camera_stream = cv2.VideoCapture(camera_url)
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

    while True:
        success, frame = camera_stream.read()
        if not success:
            break
        else:
            frame = mediapipe_process(frame, pose)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    pose.close()

@app.route('/camera/1/stream')
def camera_1_stream():
    droid_cam_url = 'http://10.1.1.211:4747/video'
    return Response(gen_frames(droid_cam_url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera/2/stream')
def camera_2_stream():
    mac_cam_url = 'http://10.1.1.242:5001/stream'
    return Response(gen_frames(mac_cam_url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
