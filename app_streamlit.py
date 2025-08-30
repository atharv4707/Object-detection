import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained emotion model
model = load_model("emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

st.title("🎥 Real-Time Emotion Detection")
st.write("Webcam-based live emotion recognition using CNN")

# Button to start webcam
if st.button("Start Webcam"):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Failed to open webcam. Please check your camera access.")
    else:
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to grab frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = model.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]

                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            # Convert frame to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")

        cap.release()
