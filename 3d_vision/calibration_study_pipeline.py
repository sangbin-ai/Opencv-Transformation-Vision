import cv2
import glob
import numpy as np
import os

CHECKERBOARD = (9, 7)

INPUT_DIR = "calib_images"
OUTPUT_DIR = "calibration_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

images = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))

# =========================================================
# 1. Object Points 준비
# =========================================================

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

object_points = []
image_points = []

image_size = None


# =========================================================
# 2. 모든 이미지에서 코너 검출
# =========================================================

for filename in images:

    image = cv2.imread(filename)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    image_size = gray.shape[::-1]

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD
    )

    if not found:
        print(f"{filename}: 코너 검출 실패")
        continue

    refined = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
    )

    object_points.append(objp.copy())
    image_points.append(refined.copy())


print("사용된 이미지 수:", len(object_points))
print("이미지 크기:", image_size)


# =========================================================
# 3. Camera Calibration
# =========================================================

rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points,
    image_points,
    image_size,
    None,
    None
)


# =========================================================
# 4. 결과 출력
# =========================================================

print("\n==============================")
print("Calibration RMS Error")
print("==============================")
print(rms)

print("\n==============================")
print("Camera Matrix K")
print("==============================")
print(camera_matrix)

print("\n==============================")
print("Distortion Coefficients")
print("==============================")
print(dist_coeffs)

print("\n==============================")
print("Intrinsic Parameters")
print("==============================")

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

print(f"fx = {fx:.4f}")
print(f"fy = {fy:.4f}")
print(f"cx = {cx:.4f}")
print(f"cy = {cy:.4f}")


# =========================================================
# 5. 텍스트 파일로 저장
# =========================================================

result_path = os.path.join(
    OUTPUT_DIR,
    "calibration_result.txt"
)

with open(result_path, "w") as f:

    f.write("Calibration RMS Error\n")
    f.write(str(rms))
    f.write("\n\n")

    f.write("Camera Matrix K\n")
    f.write(str(camera_matrix))
    f.write("\n\n")

    f.write("Distortion Coefficients\n")
    f.write(str(dist_coeffs))
    f.write("\n\n")

    f.write(f"fx = {fx}\n")
    f.write(f"fy = {fy}\n")
    f.write(f"cx = {cx}\n")
    f.write(f"cy = {cy}\n")


# =========================================================
# 6. 왜곡 보정 이미지 생성
# =========================================================

UNDISTORT_DIR = os.path.join(
    OUTPUT_DIR,
    "undistorted"
)

os.makedirs(
    UNDISTORT_DIR,
    exist_ok=True
)

for filename in images:

    image = cv2.imread(filename)

    h, w = image.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (w, h),
        1,
        (w, h)
    )

    undistorted = cv2.undistort(
        image,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix
    )

    basename = os.path.basename(filename)

    output_path = os.path.join(
        UNDISTORT_DIR,
        basename
    )

    cv2.imwrite(
        output_path,
        undistorted
    )


print("\n캘리브레이션 완료")
print("결과 저장 폴더:", OUTPUT_DIR)