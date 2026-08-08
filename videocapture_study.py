import cv2
import os

cap = cv2.VideoCapture(0)

os.makedirs("calib_images", exist_ok=True)
count = 0

while True:
    ret, frame = cap.read()

    cv2.imshow("camera", frame)

    if not ret:
        print("카메라 영상을 읽지 못했습니다.")
        break

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        filename = f"calib_images/img_{count:02d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"저장 완료: {filename}")
        count += 1

    elif key == ord('q'):
        break

    if cv2.getWindowProperty("camera", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()