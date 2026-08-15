# OpenCV Transformation Vision Study

OpenCV와 NumPy를 활용해 Computer Vision과 3D Vision의 핵심 개념을 직접 구현하여 단계적 학습을 할 예정입니다.

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
```

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
```

---

## 2 주차 - Coordinate Transformation / Convolution Hand-coding

### 1. Coordinate Transformation

```text
ArUco Pose
↓
tvec + rvec
↓
Euler Angle → Rotation Matrix
↓
Rotation + Translation
↓
Homogeneous Transformation Matrix
↓
Marker → Camera
↓
Inverse Transform
↓
Camera → Marker
↓
World → Marker 관계와 결합
↓
T_world_camera
↓
Camera Position in World
```

* `tvec`와 회전 정보를 이용해 4×4 Transformation Matrix를 구성했습니다.
* `T_camera_marker`의 역행렬을 이용해 변환 방향을 `Camera → Marker`로 변경했습니다.
* World 좌표계에서 알고 있는 ArUco Marker의 위치/방향과 결합하여 `T_world_camera`를 계산했습니다.
* 최종 변환행렬을 통해 Camera 좌표계의 점을 World 좌표계로 변환할 수 있습니다.

### 2. Convolution Hand-coding

```text
Input Image
↓
Image Patch 추출
↓
3×3 Kernel과 위치별 곱셈
↓
모든 값 합산
↓
Output의 한 좌표에 저장
↓
Kernel을 한 칸 이동
↓
반복
↓
Output Matrix
```

Padding 없이 Stride = 1일 때 Output 크기는 다음과 같이 계산했습니다.

```text
output_height = image_height - filter_height + 1
output_width  = image_width  - filter_width  + 1
```

예를 들어,

```text
Input Image : 320 × 320
Kernel      : 3 × 3

320 - 3 + 1 = 318

Output      : 318 × 318
```

NumPy의 Convolution 함수를 직접 사용하는 대신 반복문과 배열 연산으로 내부 동작을 구현했습니다.

```text
현재 (x, y) 위치
↓
Filter와 동일한 크기의 Image Patch 추출
↓
Patch × Kernel
↓
np.sum()
↓
하나의 Convolution 값 생성
↓
output[y, x]에 저장
```

이를 통해 Convolution이 이미지 전체를 한 번에 처리하는 연산이 아니라, 작은 Kernel을 이동시키면서 각 위치의 결과값을 하나씩 생성하는 과정임을 확인했습니다.

