import cv2
from pathlib import Path

RTSP_URL = "rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=1.sdp"

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def capture(name, save_dir="dataset_5g", target_size=(160,160), max_images=200):
    save_path = Path(save_dir) / name
    ensure_dir(save_path)

    cam = cv2.VideoCapture(RTSP_URL)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera at: {RTSP_URL}")

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = len(list(save_path.glob("*.jpg")))
    print(f"Connected to camera. Starting capture for '{name}'.")
    print(f"Press SPACE to capture, 'q' to quit.")

    while count < max_images:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame. Check camera connection.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show live count on frame
        cv2.putText(frame, f"Saved: {count}/{max_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Capture - Press SPACE to save face, q to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if len(faces) == 0:
                print("No face detected. Try again.")
                continue
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, target_size)
            filename = save_path / f"img_{count:05d}.jpg"
            cv2.imwrite(str(filename), face_img)
            count += 1
            print(f"Saved {filename} ({count}/{max_images})")

        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Finished. Collected {count} images for '{name}'")


if __name__ == "__main__":
    person = input("Enter person name (no spaces recommended): ").strip()
    capture(person, max_images=200)