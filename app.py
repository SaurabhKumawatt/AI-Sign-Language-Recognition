from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import threading
from gtts import gTTS
from playsound import playsound
import pygame
import os
import uuid

app = Flask(__name__, template_folder="webapp/templates")

# =========================
# 🔊 TEXT TO SPEECH SETUP
# =========================
engine = pyttsx3.init()
engine.setProperty('rate', 150)

engine_lock = threading.Lock()

# init once
pygame.mixer.init()

def speak_text(text):
    try:
        filename = f"temp_{uuid.uuid4().hex}.mp3"

        tts = gTTS(text=text, lang='en')
        tts.save(filename)

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        # wait until finished
        while pygame.mixer.music.get_busy():
            continue

        pygame.mixer.music.unload()
        os.remove(filename)

    except Exception as e:
        print("TTS Error:", e)
# def speak_text(text):
#     try:
#         with engine_lock:
#             engine.stop()
#             engine.say(text)
#             engine.runAndWait()
#     except Exception as e:
#         print("TTS Error:", e)


# =========================
# 🧠 STATE VARIABLES
# =========================
last_spoken = ""
stable_gesture = ""
stable_count = 0


# =========================
# 🤖 LOAD MODEL
# =========================
try:
    model = pickle.load(open("model/gesture_model.pkl", "rb"))
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None


# =========================
# ✋ MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)


# =========================
# 🎥 CAMERA INIT
# =========================
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Camera not accessible")


# =========================
# 🎥 FRAME GENERATOR
# =========================
def generate_frames():
    global last_spoken, stable_gesture, stable_count

    while True:
        try:
            success, frame = camera.read()

            if not success or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)

            gesture = "No Hand"
            confidence = 0

            if result.multi_hand_landmarks and model is not None:

                for hand_landmarks in result.multi_hand_landmarks:

                    # =========================
                    # 🔥 NORMALIZATION
                    # =========================
                    landmarks = []

                    base_x = hand_landmarks.landmark[0].x
                    base_y = hand_landmarks.landmark[0].y

                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x - base_x)
                        landmarks.append(lm.y - base_y)

                    landmarks = np.array(landmarks)

                    max_value = np.max(np.abs(landmarks))
                    if max_value != 0:
                        landmarks = landmarks / max_value

                    data = landmarks.reshape(1, -1)

                    # =========================
                    # 🔮 PREDICTION
                    # =========================
                    try:
                        prediction = model.predict(data)[0]

                        if hasattr(model, "predict_proba"):
                            confidence = np.max(model.predict_proba(data)) * 100

                        if confidence < 60:
                            gesture = "Unknown"
                        else:
                            gesture = prediction

                    except Exception as e:
                        print("Prediction Error:", e)
                        gesture = "Error"

                    # =========================
                    # 🔊 STABLE VOICE SYSTEM (FINAL FIX)
                    # =========================
                    valid = gesture not in ["Unknown", "No Hand", "Error"]

                    if gesture == stable_gesture:
                        stable_count += 1
                    else:
                        stable_gesture = gesture
                        stable_count = 0

                    # Speak only when stable
                    if valid and stable_count > 5 and gesture != last_spoken:
                        threading.Thread(
                            target=speak_text,
                            args=(gesture,),
                            daemon=True
                        ).start()

                        last_spoken = gesture

                    # =========================
                    # 📦 BOUNDING BOX
                    # =========================
                    h, w, _ = frame.shape

                    x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]

                    x_min, x_max = min(x_list), max(x_list)
                    y_min, y_max = min(y_list), max(y_list)

                    cv2.rectangle(
                        frame,
                        (x_min - 20, y_min - 20),
                        (x_max + 20, y_max + 20),
                        (0, 255, 0),
                        2
                    )

                    # =========================
                    # 📝 TEXT
                    # =========================
                    text_y = max(30, y_min - 30)

                    cv2.putText(
                        frame,
                        f"{gesture} ({confidence:.1f}%)",
                        (x_min - 10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

            # =========================
            # 📡 STREAM
            # =========================
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print("Frame Error:", e)
            continue


# =========================
# 🌐 ROUTES
# =========================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# =========================
# 🚀 RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)