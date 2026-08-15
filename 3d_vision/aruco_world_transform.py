import numpy as np


# =========================================================
# 1. 18번 사진에서 이미 구해놓은 ArUco Pose
#    Marker -> Camera
# =========================================================

t_camera_marker = np.array([
    0.012,   # X = 1.2 cm
    0.059,   # Y = 5.9 cm
    0.208    # Z = 20.8 cm
])


roll  = np.deg2rad(-140.5)
pitch = np.deg2rad(26.3)
yaw   = np.deg2rad(97.0)


# =========================================================
# 2. Euler Angle -> Rotation Matrix
# =========================================================

Rx = np.array([
    [1, 0, 0],
    [0, np.cos(roll), -np.sin(roll)],
    [0, np.sin(roll),  np.cos(roll)]
])

Ry = np.array([
    [ np.cos(pitch), 0, np.sin(pitch)],
    [0,              1, 0],
    [-np.sin(pitch), 0, np.cos(pitch)]
])

Rz = np.array([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw),  np.cos(yaw), 0],
    [0,            0,           1]
])


R_camera_marker = Rz @ Ry @ Rx


# =========================================================
# 3. Marker -> Camera 변환행렬
# =========================================================

T_camera_marker = np.eye(4)

T_camera_marker[:3, :3] = R_camera_marker
T_camera_marker[:3, 3] = t_camera_marker


# =========================================================
# 4. World에서 ArUco 위치/방향
# =========================================================
#
# World:
# +X = 오른쪽
# +Y = 북쪽
# +Z = 위
#
# ArUco 중심 = (170, -35, 65) cm

marker_world_position = np.array([
    1.70,
   -0.35,
    0.65
])


# ArUco 정면이 북쪽(+Y)을 바라본다고 가정
#
# Marker X -> World -X
# Marker Y -> World +Z
# Marker Z -> World +Y

R_world_marker = np.array([
    [-1, 0, 0],
    [ 0, 0, 1],
    [ 0, 1, 0]
])


T_world_marker = np.eye(4)

T_world_marker[:3, :3] = R_world_marker
T_world_marker[:3, 3] = marker_world_position


# =========================================================
# 5. Camera -> World
# =========================================================

T_marker_camera = np.linalg.inv(
    T_camera_marker
)

T_world_camera = (
    T_world_marker
    @ T_marker_camera
)


# =========================================================
# 6. World 기준 Camera 위치
# =========================================================

camera_world = T_world_camera[:3, 3]


print("Camera World Position [cm]")

print("X =", camera_world[0] * 100)
print("Y =", camera_world[1] * 100)
print("Z =", camera_world[2] * 100)


print("\nT_world_camera")

print(T_world_camera)