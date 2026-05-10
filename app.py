from time import time

from flask import Flask, render_template, Response, request, redirect, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
from gtts import gTTS
import pygame
import os
import uuid
import pandas as pd
import subprocess
import sys
import glob

training_complete = False
last_collect_time = 0

collect_mode = False
collect_gesture = ""
collect_samples = []
sample_count = 0
MAX_SAMPLES = 100



CUSTOM_DATASET_PATH = "dataset/custom"

os.makedirs(CUSTOM_DATASET_PATH, exist_ok=True)

gesture_history = []
app = Flask(__name__, template_folder="webapp/templates")

# =========================
# 🔊 TEXT TO SPEECH SETUP
# =========================

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
            pygame.time.Clock().tick(10)

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
    max_num_hands=2,
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
    global last_spoken
    global stable_gesture
    global stable_count

    global collect_mode
    global collect_samples
    global sample_count
    global collect_gesture

    global model
    global training_complete
    global last_collect_time

    while True:
        try:
            success, frame = camera.read()

            if not success or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)
            if result.multi_hand_landmarks:

                result.multi_hand_landmarks = sorted(
                    result.multi_hand_landmarks,
                    key=lambda hand: hand.landmark[0].x
    )

            gesture = "No Hand"
            confidence = 0

            if result.multi_hand_landmarks and model is not None:

                text_x = 20
                text_y = 40

                # =========================
                # 🔥 NORMALIZATION
                # =========================
                # =========================
                # ✋ TWO HAND FEATURE EXTRACTION
                # =========================

                all_landmarks = []

                # process max 2 hands
                for hand in result.multi_hand_landmarks[:2]:

                    hand_points = []

                    base_x = hand.landmark[0].x
                    base_y = hand.landmark[0].y

                    for lm in hand.landmark:

                        hand_points.append(lm.x - base_x)
                        hand_points.append(lm.y - base_y)

                    hand_points = np.array(hand_points)

                    max_value = np.max(np.abs(hand_points))

                    if max_value != 0:
                        hand_points = hand_points / max_value

                    all_landmarks.extend(hand_points.tolist())

                # If only one hand detected
                while len(all_landmarks) < 84:
                    all_landmarks.append(0)

                landmarks = np.array(all_landmarks)

                data = landmarks.reshape(1, -1)

                for hand_landmarks in result.multi_hand_landmarks:

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
                # 📸 CUSTOM GESTURE COLLECTION
                # =========================

                if collect_mode:

                    current_time = time()

                    if current_time - last_collect_time > 0.15:

                        collect_samples.append(landmarks.tolist())
                        sample_count += 1

                        last_collect_time = current_time

                    cv2.putText(
                        frame,
                        f"Collecting {collect_gesture}: {sample_count}/{MAX_SAMPLES}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

                    # Save automatically
                    if sample_count >= MAX_SAMPLES:

                        formatted_samples = []

                        for sample in collect_samples:

                            sample.append(collect_gesture)
                            formatted_samples.append(sample)

                        df = pd.DataFrame(formatted_samples)

                        save_path = os.path.join(
                            CUSTOM_DATASET_PATH,
                            f"{collect_gesture}.csv"
                        )

                        # append if already exists
                        if os.path.exists(save_path):

                            old_df = pd.read_csv(save_path)
                            df = pd.concat([old_df, df])

                        df.to_csv(save_path, index=False)

                        print(f"✅ Saved custom gesture: {collect_gesture}")

                        collect_mode = False

                        # =========================
                        # 🤖 AUTO RETRAIN
                        # =========================
                        try:
                            subprocess.run(
                                    [sys.executable, "train_model.py"],
                                    check=True
                            )

                            # Reload model

                            model = pickle.load(
                                open("model/gesture_model.pkl", "rb")
                            )

                            print("🚀 Model retrained successfully")
                            training_complete = True
                            sample_count = 0
                            collect_samples = []
                            collect_gesture = ""

                        except Exception as e:
                            print("Retrain Error:", e)

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
                    # 🕘 GESTURE HISTORY
                    # =========================
                    if (
                        gesture not in ["Unknown", "No Hand", "Error"]
                        and (len(gesture_history) == 0 or gesture != gesture_history[0])
                    ):
                        gesture_history.insert(0, gesture)

                # Keep only last 5
                gesture_history[:] = gesture_history[:5]

                

                # =========================
                # 📝 TEXT
                # =========================
                text_y = 40

                cv2.putText(
                    frame,
                    f"{gesture} ({confidence:.1f}%)",
                    (text_x, text_y),
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

    gestures = []

    # Load gestures from both folders
    csv_files = glob.glob("dataset/*.csv")
    csv_files += glob.glob("dataset/custom/*.csv")

    for file in csv_files:

        name = os.path.basename(file)
        name = name.replace(".csv", "")

        gestures.append(name.title())

    # remove duplicates
    gestures = list(set(gestures))

    # sort alphabetically
    gestures.sort()

    return render_template(
        'index.html',
        history=gesture_history,
        gestures=gestures
    )

@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/history')
def history():
    return jsonify(gesture_history)

@app.route('/add_gesture', methods=['GET', 'POST'])
def add_gesture():

    global collect_mode
    global collect_gesture
    global collect_samples
    global sample_count

    if request.method == 'POST':

        collect_gesture = request.form['gesture'].lower()

        collect_mode = True
        collect_samples = []
        sample_count = 0

        return render_template(
            'training.html',
            gesture=collect_gesture
        )

    return render_template('add_gesture.html')

@app.route('/training_status')
def training_status():

    global training_complete

    if training_complete:
        training_complete = False
        return jsonify({"done": True})

    return jsonify({"done": False})


# =========================
# 🚀 RUN
# =========================
if __name__ == "__main__":
    try:
        app.run(debug=True)

    finally:
        camera.release()
        cv2.destroyAllWindows()