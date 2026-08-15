import cv2
import os

OUTPUT_DIR = "aruco_marker"
os.makedirs(OUTPUT_DIR, exist_ok=True)

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

marker_id = 0
marker_size = 800

marker = cv2.aruco.generateImageMarker(
    dictionary,
    marker_id,
    marker_size
)

output_path = os.path.join(
    OUTPUT_DIR,
    "aruco_id_0.png"
)

cv2.imwrite(output_path, marker)

print("저장 완료:", output_path)