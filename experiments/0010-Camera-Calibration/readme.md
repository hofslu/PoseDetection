# 📷 Camera Calibration Experiment

## 🧠 Description

This experiment explores stereo camera calibration using two ESP32-CAM devices. The primary goal is to calculate the relative rotation and translation between two camera viewpoints in a minimal, user-guided setup. Instead of a traditional chessboard calibration, we rely on a well-known physical scene (e.g., a rectangular desk) and a human T-pose snapshot to verify and eventually reconstruct 3D human pose data.

## 🎯 Goals

- Estimate relative pose (rotation + translation) between two ESP32-CAMs.
- Validate calibration with human T-pose pose estimation.
- Use known scene geometry (rectangular desk) for initial correspondence points.
- Lay the foundation for full 3D human pose triangulation.
- Build and update a living documentation for reproducibility and future improvements.

## 🛣️ Roadmap

1. 📸 Capture synchronized images from both ESP32-CAMs:
   - Empty scene (room with known desk).
   - Scene with human in T-pose.
2. 🎯 Manually mark known desk points in both images.
3. 📐 Estimate relative rotation & translation using OpenCV tools:
   - `cv2.findEssentialMat` + `cv2.recoverPose`
   - OR full `cv2.stereoCalibrate` pipeline.
4. 🤖 Detect 2D keypoints in T-pose using existing pose model.
5. 📍 Match keypoints between views and triangulate 3D position.
6. 🧪 Visualize and validate 3D output.
7. 🔁 Iterate, improve robustness, and track progress below.

---

# 🛠 Documentation

## 2025-04-10 initial-setup-arno

- 🧪 Created experiment folder: `experiments/0010-Camera-Calibration`
- 📝 Initialized README.md with project goals, description, and roadmap.
- 🧠 Discussed plan to use desk and T-pose snapshots for initial 3D calibration.
- 🧰 Wrote minimal click-based image point selector and `cv2.triangulatePoints` starter logic.
- 🚀 Ready to start collecting calibration image data!
