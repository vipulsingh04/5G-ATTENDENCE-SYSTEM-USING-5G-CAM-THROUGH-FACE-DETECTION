from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
import os
import requests
# from zoneinfo import ZoneInfo
from datetime import datetime

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# ================= CONFIG =================
id_mapper = {
    "Akshat": 1001,
    "Ravi": 1002,
    "Vipul": 5775,
    "Rohan": 1010
}

API_BASE_URL = "https://fiveg-attendance-apis.onrender.com/"
ATTENDANCE_URL = f"{API_BASE_URL}/api/add-attendance"


from datetime import datetime, timezone, timedelta

def get_ist_timestamp() -> str:
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).isoformat(timespec="seconds")
print("IST TIME:", get_ist_timestamp())

def send_attendance_post(person_name: str) -> dict:
    if person_name not in id_mapper:
        return {"sent": False, "reason": "name not found in id_mapper"}

    payload = {
        "model_id": id_mapper[person_name],
        "date": get_ist_timestamp()
    }

    try:
        response = requests.post(
            ATTENDANCE_URL,
            json=payload,
            timeout=10
        )
        return {
            "sent": True,
            "status_code": response.status_code,
            "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
        }
    except Exception as e:
        return {"sent": False, "error": str(e)}
CAMERA_SOURCE = "webcam"   # "webcam" or "5g"
WEBCAM_INDEX = 0

RTSP_URL = "rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=1.sdp"

MODEL_PATH = "face_recognition_model.tflite"
CLASS_NAMES_PATH = Path("class_names.json")

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70
FACE_MIN_SIZE = (60, 60)

STREAM_TIMEOUT = 10

# ================= GLOBAL STATE =================

state: dict[str, Any] = {}
_infer_lock = threading.Lock()


# ================= MODEL LOADERS =================

def _load_tflite(path: str):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    state["backend"] = "tflite"
    state["interpreter"] = interpreter
    state["input_details"] = interpreter.get_input_details()
    state["output_details"] = interpreter.get_output_details()
    state["input_dtype"] = interpreter.get_input_details()[0]["dtype"]

    print(f"✅ TFLite model loaded from: {path}")


def _load_keras(path: str):
    import tensorflow as tf

    model = tf.keras.models.load_model(path)
    state["backend"] = "keras"
    state["model"] = model

    print(f"✅ Keras model loaded from: {path}")


# ================= LIFESPAN =================

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"class_names.json not found at: {CLASS_NAMES_PATH}")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        state["class_names"] = json.load(f)

    print(f"✅ Loaded {len(state['class_names'])} classes: {state['class_names']}")

    model_path = str(MODEL_PATH)
    if model_path.endswith(".tflite"):
        _load_tflite(model_path)
    elif model_path.endswith(".keras") or model_path.endswith(".h5"):
        _load_keras(model_path)
    else:
        for candidate in ["face_recognition_model.tflite", "model.keras", "model.h5"]:
            if Path(candidate).exists():
                if candidate.endswith(".tflite"):
                    _load_tflite(candidate)
                else:
                    _load_keras(candidate)
                break
        else:
            raise FileNotFoundError(
                "No model file found. Place face_recognition_model.tflite or model.keras here."
            )

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    state["cascade"] = cv2.CascadeClassifier(cascade_path)
    print("✅ Haar Cascade loaded")

    reader = CameraReader()
    reader.start()
    state["camera_reader"] = reader

    yield

    reader.stop()
    state.clear()
    print("🔴 Shutdown complete")


# ================= FASTAPI APP =================

app = FastAPI(
    title="Face Recognition API",
    description="Camera + face detection + recognition + live annotated preview",
    version="1.0.0",
    lifespan=lifespan,
)


# ================= CAMERA READER =================

class CameraReader:
    def __init__(self):
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        source = f"webcam #{WEBCAM_INDEX}" if CAMERA_SOURCE == "webcam" else RTSP_URL
        print(f"📷 Opening camera: {source}")

        if CAMERA_SOURCE == "webcam":
            self._cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self._cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {source}")

        # Warm-up
        warmed = False
        for _ in range(30):
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                warmed = True
                break
            time.sleep(0.1)

        if warmed:
            print("✅ Warm-up frame received")
        else:
            print("⚠️ Camera opened, but no warm-up frame yet")

        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="CameraReader")
        self._thread.start()
        print(f"✅ Camera thread started: {source}")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        print("🔴 Camera released")

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _read_loop(self):
        print("🎥 Camera read loop running")
        while not self._stop.is_set():
            if self._cap is None:
                time.sleep(0.05)
                continue

            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.05)

            time.sleep(0.02)


# ================= INFERENCE HELPERS =================

def grab_frame() -> np.ndarray:
    reader: CameraReader | None = state.get("camera_reader")
    if reader is None:
        raise RuntimeError("Camera reader is not initialised.")

    deadline = time.time() + 10
    while time.time() < deadline:
        frame = reader.get_frame()
        if frame is not None:
            return frame
        time.sleep(0.05)

    raise RuntimeError("Camera is open but no frame received yet.")


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    face_f32 = face_resized.astype(np.float32) / 255.0
    return np.expand_dims(face_f32, axis=0)


def run_inference(face_bgr: np.ndarray) -> tuple[str, float]:
    batch = preprocess_face(face_bgr)
    class_names: list[str] = state["class_names"]

    if state["backend"] == "tflite":
        interp = state["interpreter"]
        input_details = state["input_details"]
        output_details = state["output_details"]
        input_dtype = state["input_dtype"]

        if input_dtype == np.uint8:
            input_data = (batch * 255).astype(np.uint8)
        else:
            input_data = batch

        with _infer_lock:
            interp.set_tensor(input_details[0]["index"], input_data)
            interp.invoke()
            probs = interp.get_tensor(output_details[0]["index"])[0]
    else:
        probs = state["model"].predict(batch, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    name = class_names[pred_idx]
    return name, confidence


def detect_and_recognize(frame: np.ndarray, threshold: float = CONFIDENCE_THRESHOLD) -> dict | None:
    cascade: cv2.CascadeClassifier = state["cascade"]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=FACE_MIN_SIZE,
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_crop = frame[y:y + h, x:x + w]

    name, confidence = run_inference(face_crop)

    if confidence < threshold:
        name = "Unknown"

    return {
        "person_id": name,
        "name": name,
        "confidence": round(confidence * 100, 2),
        "threshold": round(threshold * 100, 2),
        "face_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "unknown": name == "Unknown",
    }


def annotate_frame(frame: np.ndarray, result: dict | None) -> np.ndarray:
    annotated = frame.copy()

    if result is None:
        cv2.putText(
            annotated,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    box = result["face_box"]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]

    color = (0, 255, 0) if not result["unknown"] else (0, 0, 255)

    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

    label = f'{result["name"]} ({result["confidence"]}%)'
    cv2.putText(
        annotated,
        label,
        (x, max(25, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    return annotated


# ================= ROUTES =================

@app.get("/")
def root():
    return {
        "service": "Face Recognition API",
        "status": "running",
        "routes": {
            "GET /health": "Check status and loaded model info",
            "GET /recognize": "Capture one frame and identify face",
            "GET /video_feed": "Live annotated MJPEG stream",
            "GET /view": "Browser page with live annotated video",
            "GET /recognize/stream": "Poll camera until a confident match (or timeout)",
        },
    }


@app.get("/health")
def health():
    reader: CameraReader | None = state.get("camera_reader")
    return {
        "status": "ok",
        "backend": state.get("backend"),
        "classes": state.get("class_names"),
        "threshold": CONFIDENCE_THRESHOLD,
        "camera_source": CAMERA_SOURCE,
        "camera_detail": f"webcam #{WEBCAM_INDEX}" if CAMERA_SOURCE == "webcam" else RTSP_URL,
        "camera_live": reader is not None and reader._frame is not None,
    }


@app.get("/recognize")
def recognize(
    threshold: float = Query(
        default=CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Confidence threshold (0–1). Override per-request.",
    )
):
    try:
        frame = grab_frame()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    result = detect_and_recognize(frame, threshold=threshold)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No face detected in the captured frame. Try again.",
        )

    # Send attendance only if a known person is detected
    attendance_result = None
    if not result["unknown"] and result["name"] in id_mapper:
        attendance_result = send_attendance_post(result["name"])

    return JSONResponse(content={
        "recognition": result,
        "attendance": attendance_result
    })


@app.get("/recognize/stream")
def recognize_stream(
    timeout: float = Query(
        default=STREAM_TIMEOUT,
        ge=1.0,
        le=60.0,
        description="How many seconds to keep trying for a confident match.",
    ),
    threshold: float = Query(
        default=CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Confidence threshold (0–1).",
    ),
):
    deadline = time.time() + timeout
    best: dict | None = None
    best_conf = 0.0

    while time.time() < deadline:
        try:
            frame = grab_frame()
        except RuntimeError:
            time.sleep(0.3)
            continue

        result = detect_and_recognize(frame, threshold=threshold)

        if result is None:
            time.sleep(0.2)
            continue

        if result["confidence"] > best_conf:
            best = result
            best_conf = result["confidence"]

        if not result["unknown"]:
            return JSONResponse(content=result)

        time.sleep(0.2)

    if best is None:
        raise HTTPException(status_code=404, detail=f"No face detected within {timeout}s timeout.")

    return JSONResponse(content=best)


@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            try:
                frame = grab_frame()
            except RuntimeError:
                time.sleep(0.1)
                continue

            result = detect_and_recognize(frame, threshold=CONFIDENCE_THRESHOLD)
            annotated = annotate_frame(frame, result)

            ok, buffer = cv2.imencode(".jpg", annotated)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(0.03)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/view", response_class=HTMLResponse)
def view():
    return """
    <!doctype html>
    <html>
    <head>
        <title>Live Face Detection</title>
        <style>
            body {
                margin: 0;
                background: #111;
                color: #fff;
                font-family: Arial, sans-serif;
                text-align: center;
            }
            h2 {
                margin: 20px 0 10px;
            }
            img {
                width: 90vw;
                max-width: 900px;
                border: 3px solid #444;
                border-radius: 12px;
                margin-bottom: 20px;
            }
            .hint {
                color: #bbb;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <h2>Live Face Detection</h2>
        <img src="/video_feed" />
        <div class="hint">Open /recognize for JSON output. This page shows the detected face box and name.</div>
    </body>
    </html>
    """


# ================= ENTRY POINT =================

if __name__ == "__main__":
    uvicorn.run("face_recognition_api:app", host="127.0.0.1", port=8000, reload=False)