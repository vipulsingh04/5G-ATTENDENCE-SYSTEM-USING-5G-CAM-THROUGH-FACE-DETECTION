import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

for i in range(50):
    ret, frame = cap.read()
    print("read:", ret, "frame is None:", frame is None)
    if ret and frame is not None:
        cv2.imshow("camera test", frame)
        cv2.waitKey(1000)
        break
    time.sleep(0.1)

cap.release()
cv2.destroyAllWindows()