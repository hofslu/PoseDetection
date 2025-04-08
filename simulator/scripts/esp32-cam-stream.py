from datetime import datetime
import pygame
import requests
import cv2
import mediapipe as mp
import dotenv
from collections import deque
import json
import numpy as np
import time
import os
from io import BytesIO

# Load environment variables from .env file
dotenv.load_dotenv()

WHATSAPP_API_KEY = os.getenv("WHATSAPP_API_KEY")


# 🧠 Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# 🖼️ Settings
face_detection_enabled = False
pose_detection_enabled = True

# 🌐 ESP32-CAM stream URL
ESP32_URL = 'http://10.1.1.166/stream'
show_stats = False
stats_window = None

up_pose_data = None
down_pose_data = None
pushup_count = 0
last_pose_state = None  # 'up', 'down', or None
pose_capture_countdown = 0
capture_target = None  # 'up' or 'down'
capture_start_time = 0



# History buffers (10 sec assuming 30 fps ≈ 300 samples)
MAX_SAMPLES = 300
fetch_times = deque(maxlen=MAX_SAMPLES)
process_times = deque(maxlen=MAX_SAMPLES)
comparison_times = deque(maxlen=MAX_SAMPLES)
frame_dts = deque(maxlen=MAX_SAMPLES)
similarities = deque(maxlen=MAX_SAMPLES)


# 🖼️ Pygame setup
pygame.init()
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("ESP32-CAM Pose Viewer")
font = pygame.font.SysFont('Courier', 18)
pygame.mixer.init()
coin_sound = pygame.mixer.Sound("coin.wav")  # you can use any short WAV
power_up_sound = pygame.mixer.Sound("power_up.wav")  # you can use any short WAV


# 🕒 Time trackers
prev_frame_time = time.time()
fetch_time = process_time = fps = comparison_time = 0
paused = False

# 🧍‍♂️ Pose capture paths
t_pose_path = "t_pose.npy"
t_pose_data = None
up_pose_path = "pushup_up_pose.npy"
down_pose_path = "pushup_down_pose.npy"
rep_log_file = f"pushup_log_{datetime.now().strftime('%Y%m%d')}.json"
rep_log = []


# Load if available
if os.path.exists(t_pose_path):
    t_pose_data = np.load(t_pose_path)
if os.path.exists(up_pose_path):
    up_pose_data = np.load(up_pose_path)
if os.path.exists(down_pose_path):
    down_pose_data = np.load(down_pose_path)
if os.path.exists(rep_log_file):
    with open(rep_log_file, "r") as f:
        rep_log = json.load(f)


# 🌪️ MJPEG stream fetcher
stream = requests.get(ESP32_URL, stream=True)
bytes_data = b''

def draw_text(surface, text, x, y, color=(255, 255, 255)):
    label = font.render(text, True, color)
    surface.blit(label, (x, y))

def get_landmark_array(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)

# List of (left_idx, right_idx) to swap
MIRROR_LANDMARKS = [
    (11, 12), (13, 14), (15, 16),  # shoulders, elbows, wrists
    (23, 24), (25, 26), (27, 28)   # hips, knees, ankles
]

def mirror_pose(pose):
    mirrored = pose.copy()
    for l, r in MIRROR_LANDMARKS:
        mirrored[l], mirrored[r] = pose[r], pose[l]
    # flip X-coordinates
    mirrored[:, 0] *= -1
    return mirrored

def pose_similarity(pose1, pose2, use_3d=False):
    if pose1.shape != pose2.shape:
        return 0.0

    def normalize(pose):
        center = (pose[11] + pose[12]) / 2
        pose -= center
        scale = np.linalg.norm(pose[11] - pose[12]) + 1e-6
        return pose / scale

    if not use_3d:
        pose1 = pose1[:, :2]
        pose2 = pose2[:, :2]

    def cosine_score(a, b):
        a_norm = normalize(a.copy())
        b_norm = normalize(b.copy())
        sims = []
        for v1, v2 in zip(a_norm, b_norm):
            dot = np.dot(v1, v2)
            sim = dot / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            sims.append(sim)
        return np.mean(sims)

    score_direct = cosine_score(pose1, pose2)
    score_mirrored = cosine_score(pose1, mirror_pose(pose2))

    return max(0.0, min(max(score_direct, score_mirrored) * 100.0, 100.0))

def draw_distribution(surface, data, label, y_offset, color=(180, 180, 100)):
    if len(data) < 2:
        return
    width = 140
    height = 80
    bins = 20
    hist, bin_edges = np.histogram(data, bins=bins, range=(min(data), max(data)))
    max_count = max(hist) + 1e-6  # avoid div by zero

    for i in range(bins):
        bin_height = int((hist[i] / max_count) * height)
        bin_x = i * (width // bins)
        pygame.draw.rect(
            surface, color,
            pygame.Rect(
                440 + bin_x, y_offset + height - bin_height,
                (width // bins) - 1, bin_height
            )
        )

    draw_text(surface, f"{label} dist", 440, y_offset - 20, color)

def draw_stats_text(surface, data, label, y_offset, color=(200, 200, 200), in_ms=True):
    if in_ms:
        data = [d * 1000 for d in data]
    if len(data) < 2:
        return
    x_offset = 700
    data_np = np.array(data)
    avg = np.mean(data_np)
    med = np.median(data_np)
    min_val = np.min(data_np)
    max_val = np.max(data_np)
    p25 = np.percentile(data_np, 25)
    p75 = np.percentile(data_np, 75)

    draw_text(surface, f"{label} stats", x_offset, y_offset - 20, color)
    draw_text(surface, f"Avg: {avg:.2f}", x_offset, y_offset, color)
    draw_text(surface, f"Med: {med:.2f}", x_offset, y_offset + 20, color)
    draw_text(surface, f"Min/Max: {min_val:.2f} / {max_val:.2f}", x_offset, y_offset + 40, color)
    draw_text(surface, f"25%/75%: {p25:.2f} / {p75:.2f}", x_offset, y_offset + 60, color)


running = True
while running:
    frame_start = time.time()

    # 📥 Fetch single JPEG frame from MJPEG stream
    try:
        chunk_start = time.time()
        bytes_data += stream.raw.read(1024)
        a = bytes_data.find(b'\xff\xd8')  # JPEG start
        b = bytes_data.find(b'\xff\xd9')  # JPEG end
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            img_np = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            fetch_time = time.time() - chunk_start

            if not paused:
                # 🧠 Pose processing
                process_start = time.time()
                rgb_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                if pose_detection_enabled:
                    pose_results = pose.process(rgb_img)
                    if pose_results.pose_landmarks:
                        pose_landmarks = pose_results.pose_landmarks
                        mp_drawing.draw_landmarks(img_np, pose_landmarks, mp_pose.POSE_CONNECTIONS)

                        # 🔍 Similarity calculation if T-pose saved
                        current_pose_array = get_landmark_array(pose_landmarks)
                        similarity_score = None
                        if t_pose_data is not None:
                            similarity_score = pose_similarity(current_pose_array, t_pose_data)
                    else:
                        similarity_score = None
                    
                if face_detection_enabled:
                    face_results = face.process(rgb_img)
                    if face_results.multi_face_landmarks:
                        for fl in face_results.multi_face_landmarks:
                            mp_drawing.draw_landmarks(
                                image=img_np,
                                landmark_list=fl,
                                connections=mp_face.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
                            )
                process_time = time.time() - process_start
                
            else:
                similarity_score = None
            
            comparison_time_start = time.time()
            pose_similarity_up = None
            pose_similarity_down = None

            if pose_landmarks:
                current_pose_array = get_landmark_array(pose_landmarks)

                if up_pose_data is not None:
                    pose_similarity_up = pose_similarity(current_pose_array, up_pose_data)

                if down_pose_data is not None:
                    pose_similarity_down = pose_similarity(current_pose_array, down_pose_data)

                # Push-up detection
                threshold = 85  # adjust if needed
                if pose_similarity_up and pose_similarity_down:
                    if pose_similarity_down > threshold and last_pose_state != 'down':
                        last_pose_state = 'down'
                    elif pose_similarity_up > threshold and last_pose_state == 'down':
                        pushup_count += 1
                        last_pose_state = 'up'
                        print(f"💪 Push-Up Count: {pushup_count}")
                        if pushup_count % 5 == 0:
                            pygame.mixer.Sound.play(power_up_sound)
                        else:
                            pygame.mixer.Sound.play(coin_sound)

                        rep_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "similarity_up": round(pose_similarity_up, 2),
                            "similarity_down": round(pose_similarity_down, 2),
                        }
                        rep_log.append(rep_entry)
                        with open(rep_log_file, "w") as f:
                            json.dump(rep_log, f, indent=2)
            comparison_time = time.time() - comparison_time_start


            # 🖥️ Convert to Pygame surface
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_surface = pygame.surfarray.make_surface(np.rot90(img_np))

            screen.blit(img_surface, (0, 0))

            # 🧾 Overlay Stats
            now = time.time()
            frame_to_frame = now - prev_frame_time
            prev_frame_time = now
            fps = 1.0 / frame_to_frame if frame_to_frame > 0 else 0

            y_offset = 10
            draw_text(screen, f"Fetch: {fetch_time*1000:.1f} ms", 10, y_offset)
            y_offset += 20
            draw_text(screen, f"Process: {process_time*1000:.1f} ms", 10, y_offset)
            y_offset += 20
            # comparison time
            draw_text(screen, f"Comparison: {comparison_time*100:.1f} ms", 10, y_offset)
            y_offset += 20
            draw_text(screen, f"Frame dt: {frame_to_frame*1000:.1f} ms", 10, y_offset)
            y_offset += 20
            draw_text(screen, f"FPS: {fps:.1f}", 10, y_offset)
            y_offset += 20
            draw_text(screen, f"Push-Ups: {pushup_count}", 10, y_offset)

            if similarity_score is not None:
                draw_text(screen, f"T-Pose Similarity: {similarity_score:.1f}%", screen_width - 280, 10, color=(0, 255, 0))

            if paused:
                draw_text(screen, "PAUSED", screen_width//2 - 50, screen_height//2, color=(255, 0, 0))

            # 🗂️ Store stats 
            fetch_times.append(fetch_time)
            process_times.append(process_time)
            comparison_times.append(comparison_time)
            frame_dts.append(frame_to_frame)
            if pose_landmarks and t_pose_data is not None:
                similarities.append(pose_similarity(get_landmark_array(pose_landmarks), t_pose_data))
            else:
                similarities.append(0)


            pygame.display.flip()

            if show_stats:
                if not stats_window:
                    stats_window = pygame.display.set_mode((1000, 600), pygame.RESIZABLE)
                    pygame.display.set_caption("📊 Real-Time Stats")

                stats_window.fill((30, 30, 30))

                def draw_hist(surface, data, label, y_offset, color=(100, 200, 100)):
                    if len(data) < 2:
                        return
                    width = surface.get_width()
                    height = 80
                    step = max(1, len(data) // width)
                    scaled = [min(1.0, d / max(data)) for d in data]
                    for i, val in enumerate(scaled[::step]):
                        x = i
                        y = int(val * height)
                        pygame.draw.line(surface, color, (x, y_offset + height), (x, y_offset + height - y))
                    draw_text(surface, f"{label} (last {len(data)} frames)", 10, y_offset - 20, color)

                # Draw histograms
                y_offset = 40
                draw_hist(stats_window, list(fetch_times), "Fetch Time", y_offset)
                y_offset += 100
                draw_hist(stats_window, list(process_times), "Process Time", y_offset)
                y_offset += 100
                draw_hist(stats_window, list(comparison_times), "Comparison Time", y_offset)
                y_offset += 100
                draw_hist(stats_window, list(frame_dts), "Frame dt", y_offset)
                y_offset += 100
                draw_hist(stats_window, list(similarities), "T-Pose Similarity", y_offset, color=(100, 180, 250))
                # Draw distributions
                draw_distribution(stats_window, list(fetch_times), "Fetch Time", 40)
                draw_distribution(stats_window, list(process_times), "Process Time", 140)
                draw_distribution(stats_window, list(frame_dts), "Frame dt", 240)
                draw_distribution(stats_window, list(similarities), "Similarity", 340)
                # Draw stats text
                draw_stats_text(stats_window, list(fetch_times), "Fetch Time", 40, in_ms=True)
                draw_stats_text(stats_window, list(process_times), "Process Time", 140, in_ms=True)
                draw_stats_text(stats_window, list(frame_dts), "Frame dt", 240, in_ms=True)
                draw_stats_text(stats_window, list(similarities), "Similarity", 340, in_ms=False)

                pygame.display.update()


    except Exception as e:
        print("⚠️ Error:", e)

    # 🕹️ Handle Input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                paused = not paused
                print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")

            # Pose Captures
            if event.key == pygame.K_t and not paused and pose_landmarks:
                t_pose_data = get_landmark_array(pose_landmarks)
                np.save(t_pose_path, t_pose_data)
                print("🧍‍♂️ T-Pose saved for future reference!")
            
            if event.key == pygame.K_u:
                capture_target = 'up'
                capture_start_time = time.time()
                print("⏳ Hold 'UP' position - recording in 5 seconds...")

            if event.key == pygame.K_i:
                capture_target = 'down'
                capture_start_time = time.time()
                print("⏳ Hold 'DOWN' position - recording in 5 seconds...")

            # Toggle pose and face detection
            if event.key == pygame.K_g:
                pose_detection_enabled = not pose_detection_enabled
                print(f"{'🧍‍♂️ Pose Detection ON' if pose_detection_enabled else '🧍‍♂️ Pose Detection OFF'}")

            if event.key == pygame.K_f:
                face_detection_enabled = not face_detection_enabled
                print(f"{'🙂 Face Detection ON' if face_detection_enabled else '🙂 Face Detection OFF'}")
            
            # Toggle stats window
            if event.key == pygame.K_a:
                show_stats = not show_stats
                print(f"{'📊 Stats Window Opened' if show_stats else '📉 Stats Window Closed'}")
                if not show_stats and stats_window:
                    # Restore main display after closing stats window
                    stats_window = None
                    screen = pygame.display.set_mode((screen_width, screen_height))
                    pygame.display.set_caption("ESP32-CAM Pose Viewer")
    
    # Countdown logic
    if capture_target and time.time() - capture_start_time >= 5:
        if pose_landmarks:
            captured = get_landmark_array(pose_landmarks)
            if capture_target == 'up':
                up_pose_data = captured
                np.save(up_pose_path, up_pose_data)
                print("📸 'Up' Pose Captured and Saved!")
            elif capture_target == 'down':
                down_pose_data = captured
                np.save(down_pose_path, down_pose_data)
                print("📸 'Down' Pose Captured and Saved!")
        else:
            print("❌ No pose detected during capture!")
        capture_target = None

            



pygame.quit()
stream.close()

print("📧 Send pushup_log.json to pythonanywhere/calender instance.")
try:
    if pushup_count > 0:
        print("📱 Sending WhatsApp message...")
        url = "https://holu.pythonanywhere.com/e253e0ac-9a2b-45b7-9d29-91bd1566676d"  # Replace with the actual URL
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": WHATSAPP_API_KEY
        }
        payload = {
            "msg": "Hello Lukas, you're awesome!"
        }

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print("✅ Message sent successfully!")
        else:
            print(f"❌ Failed to send message: {response.status_code} - {response.text}")
except Exception as e:
    print(f"❌ Error sending message: {e}")
finally:
    print("👋 Goodbye!")
