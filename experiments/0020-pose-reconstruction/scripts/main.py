# 🎥 Focused Stream Debug View — Dual Camera (CAM1 + CAM2)

import os
import pygame
import cv2
import numpy as np
import requests
import dotenv
from io import BytesIO

dotenv.load_dotenv()

# Stream URLs
ESP32_CAM1_URL = os.getenv("ESP32_CAM1_URL")
ESP32_CAM2_URL = os.getenv("ESP32_CAM2_URL")

# Setup Pygame window
pygame.init()
screen = pygame.display.set_mode((1280, 480))
pygame.display.set_caption("ESP32-CAM1 & CAM2 Stream Viewer")
clock = pygame.time.Clock()

# MJPEG frame fetcher
def get_mjpeg_frame(url, buffer):
    buffer += url.raw.read(1024)
    start = buffer.find(b'\xff\xd8')
    end = buffer.find(b'\xff\xd9')
    if start != -1 and end != -1:
        jpg = buffer[start:end + 2]
        buffer = buffer[end + 2:]
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img, buffer
    return None, buffer

# Connect to MJPEG streams
stream1 = requests.get(ESP32_CAM1_URL, stream=True)
stream2 = requests.get(ESP32_CAM2_URL, stream=True)
buffer1 = b""
buffer2 = b""

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fetch camera frames
    frame1, buffer1 = get_mjpeg_frame(stream1, buffer1)
    frame2, buffer2 = get_mjpeg_frame(stream2, buffer2)

    # Draw frames if valid
    if frame1 is not None:
        frame1 = cv2.resize(frame1, (640, 480))
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        surface1 = pygame.image.frombuffer(frame1.tobytes(), frame1.shape[1::-1], "RGB")
        screen.blit(surface1, (0, 0))

    if frame2 is not None:
        frame2 = cv2.resize(frame2, (640, 480))
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        surface2 = pygame.image.frombuffer(frame2.tobytes(), frame2.shape[1::-1], "RGB")
        screen.blit(surface2, (640, 0))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
