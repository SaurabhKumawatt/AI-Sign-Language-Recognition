import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load model
model = pickle.load(open("model/gesture_model.pkl", "rb"))

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "No Hand"
    confidence = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # =========================
            # 🔥 SAME NORMALIZATION
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
            # 🔮 Prediction
            # =========================
            prediction = model.predict(data)[0]

            if hasattr(model, "predict_proba"):
                confidence = np.max(model.predict_proba(data)) * 100

            # =========================
            # ⚠️ Confidence Filter
            # =========================
            if confidence < 60:
                gesture = "Unknown"
            else:
                gesture = prediction

    # =========================
    # 🖥️ Display
    # =========================
    cv2.putText(frame, f"Gesture: {gesture}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.putText(frame, f"Confidence: {confidence:.2f}%",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)

    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()