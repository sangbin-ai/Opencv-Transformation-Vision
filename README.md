# OpenCV Transformation Vision Study

OpenCV를 활용해 Camera Calibration → Pose Estimation → Coordinate Transformation 순서로 학습합니다.

---

## 1 주차 - Camera Calibration / ArUco Pose Estimation

### 1. Camera Calibration

```text
Checkerboard
↓
Corner Detection
↓
Sub-pixel Refinement
↓
Object Points ↔ Image Points
↓
cv2.calibrateCamera()
↓
K + Distortion

### 2. ArUco Pose Estimation

```text
ArUco Marker
↓
Known 3D Points ↔ Detected 2D Corners
↓
cv2.solvePnP()
↓
tvec + rvec
↓
Position + Orientation
↓
Pose
