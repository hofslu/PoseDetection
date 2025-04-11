# 🎥👓 Stream + 3D Center View (CAM1 | 3D | CAM2)

import os
import pygame
import cv2
import numpy as np
import requests
import dotenv
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
    glDisable(GL_LIGHTING)  # ✅ Disable lighting to avoid tint
    glClearColor(0.1, 0.1, 0.1, 1)

angle_x, angle_y = 0, 0

# 🔄 Reusable texture updater
# ✅ Flip the frame itself for correct orientation
def update_texture(tex_id, frame):
    frame = cv2.resize(frame, (640, 480))
    # frame = cv2.flip(frame, 1)
    frame = cv2.flip(frame, 0)
    frame_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_data)

# 🧱 Draw textured quad at a given screen location
# 👇 Coordinates no longer flipped

def draw_textured_quad(tex_id, x, width):
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glColor3f(1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(x, 0)
    glTexCoord2f(1, 1); glVertex2f(x + width, 0)
    glTexCoord2f(1, 0); glVertex2f(x + width, 480)
    glTexCoord2f(0, 0); glVertex2f(x, 480)
    glEnd()

# 🟢 Center 3D View

def draw_center_sphere():
    glBindTexture(GL_TEXTURE_2D, 0)
    glPushMatrix()
    glTranslatef(0, 0, -6)
    glColor3f(0.2, 0.7, 0.9)
    quad = gluNewQuadric()
    gluSphere(quad, 1, 32, 32)
    gluDeleteQuadric(quad)
    glPopMatrix()

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

stream1 = requests.get(ESP32_CAM1_URL, stream=True)
stream2 = requests.get(ESP32_CAM2_URL, stream=True)
buffer1 = b""
buffer2 = b""

init_opengl()
tex1 = glGenTextures(1)
tex2 = glGenTextures(1)
running = True

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        if event.type == MOUSEMOTION and pygame.mouse.get_pressed()[0]:
            dx, dy = event.rel
            angle_x += dx * 0.5
            angle_y += dy * 0.5

    frame1, buffer1 = get_mjpeg_frame(stream1, buffer1)
    frame2, buffer2 = get_mjpeg_frame(stream2, buffer2)

    if frame1 is not None:
        update_texture(tex1, frame1)
    if frame2 is not None:
        update_texture(tex2, frame2)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # --- Left Camera ---
    glViewport(0, 0, 640, 480)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 640, 480, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    draw_textured_quad(tex1, 0, 640)

    # --- Center 3D ---
    glViewport(640, 0, 640, 480)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (640 / 480), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0, 0, -10)
    glRotatef(angle_y, 1, 0, 0)
    glRotatef(angle_x, 0, 1, 0)
    draw_center_sphere()

    # --- Right Camera ---
    glViewport(1280, 0, 640, 480)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 640, 480, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    draw_textured_quad(tex2, 0, 640)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()