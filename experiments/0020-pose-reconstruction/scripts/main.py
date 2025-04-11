# 🎥👓 Stream + 3D Center View (CAM1 | 3D | CAM2)

import os
import pygame
import cv2
import numpy as np
import requests
import dotenv
import threading
from io import BytesIO
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

dotenv.load_dotenv()

ESP32_CAM1_URL = os.getenv("ESP32_CAM1_URL")
ESP32_CAM2_URL = os.getenv("ESP32_CAM2_URL")

pygame.init()
screen = pygame.display.set_mode((1920, 480), DOUBLEBUF | OPENGL)
pygame.display.set_caption("ESP32-CAM Stereo Viewer + 3D")
clock = pygame.time.Clock()

def init_opengl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glDisable(GL_LIGHTING)
    glClearColor(0.1, 0.1, 0.1, 1)

angle_x, angle_y = 0, 0

# 🔄 Reusable texture updater
def update_texture(tex_id, frame):
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 0)
    frame_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 640, 480, GL_RGB, GL_UNSIGNED_BYTE, frame_data)

# 🧱 Draw textured quad

def draw_textured_quad(tex_id, x, width):
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glColor3f(1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(x, 0)
    glTexCoord2f(1, 1); glVertex2f(x + width, 0)
    glTexCoord2f(1, 0); glVertex2f(x + width, 480)
    glTexCoord2f(0, 0); glVertex2f(x, 480)
    glEnd()

# 🟢 Center 3D Sphere

def draw_center_sphere():
    glBindTexture(GL_TEXTURE_2D, 0)
    glPushMatrix()
    glTranslatef(0, 0, -6)
    glColor3f(0.2, 0.7, 0.9)
    quad = gluNewQuadric()
    gluSphere(quad, 1, 32, 32)
    gluDeleteQuadric(quad)
    glPopMatrix()

# ⚡ Live Frame Grabbers

def mjpeg_stream_reader(url, frame_container):
    buffer = b""
    stream = requests.get(url, stream=True)
    while True:
        buffer += stream.raw.read(8192)
        start = buffer.find(b'\xff\xd8')
        end = buffer.find(b'\xff\xd9')
        if start != -1 and end != -1:
            jpg = buffer[start:end + 2]
            buffer = buffer[end + 2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                frame_container["frame"] = img

init_opengl()
tex1, tex2 = glGenTextures(2)

# Init with dummy texture
for tex in [tex1, tex2]:
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

frame1 = {"frame": None}
frame2 = {"frame": None}
threading.Thread(target=mjpeg_stream_reader, args=(ESP32_CAM1_URL, frame1), daemon=True).start()
threading.Thread(target=mjpeg_stream_reader, args=(ESP32_CAM2_URL, frame2), daemon=True).start()

running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEMOTION and pygame.mouse.get_pressed()[0]:
            dx, dy = event.rel
            angle_x += dx * 0.5
            angle_y += dy * 0.5

    if frame1["frame"] is not None:
        update_texture(tex1, frame1["frame"])
    if frame2["frame"] is not None:
        update_texture(tex2, frame2["frame"])

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # --- Left ---
    glViewport(0, 0, 640, 480)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 640, 480, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    draw_textured_quad(tex1, 0, 640)

    # --- Center ---
    glViewport(640, 0, 640, 480)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 640/480, 0.1, 1000)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0, 0, -10)
    glRotatef(angle_y, 1, 0, 0)
    glRotatef(angle_x, 0, 1, 0)
    draw_center_sphere()

    # --- Right ---
    glViewport(1280, 0, 640, 480)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 640, 480, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    draw_textured_quad(tex2, 0, 640)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()