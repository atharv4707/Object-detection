import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from collections import Counter
import time

# --- App Config ---
st.set_page_config(page_title="VisionLive: Real-Time Object Detection", page_icon="🤖", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
        .main {background-color: #0D1117;}
        h1 {
            background: linear-gradient(to right, #00FFA3, #DC1FFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-size: 2.5rem;
        }
        .card {
            padding: 15px;
            border-radius: 10px;
            background-color: #161B22;
            color: white;
            margin-bottom: 15px;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    mode = st.radio("Choose Mode", ["📸 Capture Image", "🎥 Live Webcam"])
    st.markdown("---")
    st.markdown("👨‍💻 Powered by YOLOv8 + Streamlit")

# --- Title ---
st.markdown("<h1>🤖 VisionLive: Real-Time Object Detection</h1>", unsafe_allow_html=True)
st.caption("Detect objects in real-time using YOLOv8 and your webcam.")

# --- Load Model ---
@st.cache_resource(show_spinner=True)
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# ==================================================
# MODE 1: Capture Single Image
# ==================================================
if mode == "📸 Capture Image":
    img_file_buffer = st.camera_input("📸 Take a picture")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        img_array = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        results = model(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()

        st.image(annotated_frame, channels="BGR", use_container_width=True, caption="📹 Detection Result")

        names = results[0].names
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            counts = Counter(class_ids)
            detected_summary = " | ".join([f"{names[c].capitalize()}: {counts[c]}" for c in counts])
            st.markdown(f"### 🔎 Detected: {detected_summary}")
        else:
            st.markdown("### 🔎 Detected: None ❌")

# ==================================================
# MODE 2: Live Webcam Stream
# ==================================================
elif mode == "🎥 Live Webcam":
    run = st.toggle("🎥 Start Webcam", value=False)
    FRAME_WINDOW = st.empty()
    DETECTED_WINDOW = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        st.info("✅ Webcam started. Uncheck to stop.")

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to grab frame from webcam.")
                break

            frame = cv2.flip(frame, 1)  # Mirror effect
            results = model(frame, conf=conf_threshold)
            annotated_frame = results[0].plot()

            FRAME_WINDOW.image(annotated_frame, channels="BGR", use_container_width=True, caption="📹 Live Detection")

            names = results[0].names
            boxes = results[0].boxes
            with DETECTED_WINDOW.container():
                if boxes is not None and len(boxes) > 0:
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    counts = Counter(class_ids)
                    detected_summary = " | ".join([f"{names[c].capitalize()}: {counts[c]}" for c in counts])
                    st.markdown(f"### 🔎 Detected: {detected_summary}")
                else:
                    st.markdown("### 🔎 Detected: None ❌")

            time.sleep(0.03)

        cap.release()
        st.success("🛑 Webcam stopped.")
