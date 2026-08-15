import cv2
import numpy as np
import os
import glob
import time


# =========================================================
# Camera Calibration 결과
# =========================================================

CAMERA_MATRIX = np.array([
    [502.24524145,   0.0,          327.93323318],
    [0.0,            497.13261842, 235.04246511],
    [0.0,            0.0,           1.0]
], dtype=np.float64)

DIST_COEFFS = np.array([
    -1.51666372e-02,
     1.02349149e-01,
    -8.67719450e-05,
    -2.73703081e-03,
    -1.49108386e-01
], dtype=np.float64)


# =========================================================
# ArUco 실제 크기
# =========================================================

MARKER_LENGTH = 0.0475  # 4.75 cm

half = MARKER_LENGTH / 2.0

object_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0]
], dtype=np.float32)


# =========================================================
# ArUco Detector
# =========================================================

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    dictionary,
    parameters
)


# =========================================================
# 결과 폴더
# =========================================================

OUTPUT_DIR = "aruco_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 기존 파일 번호 확인
# =========================================================

existing_files = glob.glob(
    os.path.join(
        OUTPUT_DIR,
        "aruco_pose_*.jpg"
    )
)

numbers = []

for file in existing_files:

    basename = os.path.basename(file)

    try:
        number = int(
            basename
            .replace("aruco_pose_", "")
            .replace(".jpg", "")
        )

        numbers.append(number)

    except ValueError:
        pass


if numbers:
    save_count = max(numbers) + 1
else:
    save_count = 0


print(f"저장 시작 번호: {save_count}")


# =========================================================
# 자동 촬영 설정
# =========================================================

MAX_SAVE_COUNT = 10

COOLDOWN_SECONDS = 5.0

saved_this_run = 0

# 처음 실행했을 때는 바로 촬영 가능하게
last_save_time = -COOLDOWN_SECONDS


# =========================================================
# Camera
# =========================================================

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        print("카메라 영상을 읽지 못했습니다.")
        break


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # =====================================================
    # ArUco 검출
    # =====================================================

    corners, ids, rejected = detector.detectMarkers(
        gray
    )

    marker_detected = False


    if ids is not None:

        cv2.aruco.drawDetectedMarkers(
            frame,
            corners,
            ids
        )


        for marker_corners, marker_id in zip(
            corners,
            ids
        ):

            marker_id = int(
                np.asarray(marker_id)
                .reshape(-1)[0]
            )


            image_points = (
                marker_corners
                .reshape(4, 2)
                .astype(np.float32)
            )


            # =================================================
            # Pose Estimation
            # =================================================

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                CAMERA_MATRIX,
                DIST_COEFFS,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )


            if not success:
                continue


            marker_detected = True


            # =================================================
            # XYZ
            # =================================================

            X = tvec[0][0]
            Y = tvec[1][0]
            Z = tvec[2][0]

            distance = np.linalg.norm(tvec)


            # =================================================
            # Orientation
            # rvec -> Rotation Matrix -> Euler Angle
            # =================================================

            rotation_matrix, _ = cv2.Rodrigues(rvec)

            projection_matrix = np.hstack((
                rotation_matrix,
                np.zeros((3, 1))
            ))

            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
                projection_matrix
            )

            roll = float(euler_angles[0][0])
            pitch = float(euler_angles[1][0])
            yaw = float(euler_angles[2][0])
            # =================================================
            # 좌표축 표시
            # =================================================


            cv2.drawFrameAxes(
                frame,
                CAMERA_MATRIX,
                DIST_COEFFS,
                rvec,
                tvec,
                MARKER_LENGTH * 0.5
            )


            # =================================================
            # 텍스트 표시
            # =================================================

            corner = image_points[0]

            text_x = int(corner[0])
            text_y = int(corner[1]) - 50


            cv2.putText(
                frame,
                f"ID: {marker_id}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"X: {X * 100:.1f} cm",
                (text_x, text_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Y: {Y * 100:.1f} cm",
                (text_x, text_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Z: {Z * 100:.1f} cm",
                (text_x, text_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Distance: {distance * 100:.1f} cm",
                (text_x, text_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Roll: {roll:.1f} deg",
                (text_x, text_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Pitch: {pitch:.1f} deg",
                (text_x, text_y + 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Yaw: {yaw:.1f} deg",
                (text_x, text_y + 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )

    # =====================================================
    # 즉시 자동 저장
    # =====================================================

    current_time = time.time()

    time_since_last_save = (
        current_time - last_save_time
    )


    if (
        marker_detected
        and time_since_last_save >= COOLDOWN_SECONDS
        and saved_this_run < MAX_SAVE_COUNT
    ):

        filename = os.path.join(
            OUTPUT_DIR,
            f"aruco_pose_{save_count:02d}.jpg"
        )

        # 검출된 바로 그 frame을 즉시 저장
        cv2.imwrite(
            filename,
            frame
        )

        print(
            f"즉시 자동 저장: {filename}"
        )

        save_count += 1

        saved_this_run += 1

        last_save_time = current_time


    # =====================================================
    # 상태 표시
    # =====================================================

    cv2.putText(
        frame,
        f"Saved: {saved_this_run}/{MAX_SAVE_COUNT}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    # 남은 쿨다운 표시
    remaining = max(
        0,
        COOLDOWN_SECONDS - time_since_last_save
    )

    cv2.putText(
        frame,
        f"Cooldown: {remaining:.1f}s",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    if saved_this_run >= MAX_SAVE_COUNT:

        cv2.putText(
            frame,
            "AUTO CAPTURE COMPLETE",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


    # =====================================================
    # 화면 표시
    # =====================================================

    cv2.imshow(
        "ArUco Pose Estimation",
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()