import cv2
import numpy as np

found_index = None
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        found_index = i
        cap.release()
        break

window_name = "Camera Scan"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 480)

if found_index is None:
    frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "No camera found",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    while True:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
else:
    cap = cv2.VideoCapture(found_index)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(
            frame,
            f"Camera found at index {found_index} - Press ESC to exit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()

cv2.destroyAllWindows()
