import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Test", 640, 480)

while True:
    ret, frame = cap.read()

    if not ret:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Cannot access camera",
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "Press ESC to exit",
            (20, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    cv2.putText(
        frame,
        "Camera Test - Press ESC to Exit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
