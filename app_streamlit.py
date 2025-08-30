import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load model safely
try:
    model = load_model("emotion_model.h5")
except Exception as e:
    st.error(f"Model not found or failed to load: {e}")
    st.stop()

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion labels
labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

st.title("🎥 Real-time Emotion Detection")
st.write("Detects emotions live from your webcam")

# Webcam start button
run = st.checkbox("Start Webcam")

if run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("⚠️ Failed to open webcam. Please check camera permissions.")
        st.stop()

    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Failed to read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi)[0]
            label = labels[np.argmax(preds)]

            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
