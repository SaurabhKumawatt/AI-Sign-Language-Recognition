import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

gesture_name = input("Enter gesture label: ").lower()

os.makedirs("dataset", exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

data = []

print("Press 's' to save sample")
print("Press 'q' to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        # Sort hands left → right
        result.multi_hand_landmarks = sorted(
            result.multi_hand_landmarks,
            key=lambda hand: hand.landmark[0].x
        )

        all_landmarks = []

        # =========================
        # ✋ PROCESS MAX 2 HANDS
        # =========================
        for hand_landmarks in result.multi_hand_landmarks[:2]:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            hand_points = []

            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y

            for lm in hand_landmarks.landmark:

                hand_points.append(lm.x - base_x)
                hand_points.append(lm.y - base_y)

            hand_points = np.array(hand_points)

            max_value = np.max(np.abs(hand_points))

            if max_value != 0:
                hand_points = hand_points / max_value

            all_landmarks.extend(hand_points.tolist())

        # =========================
        # ✋ PAD IF ONLY 1 HAND
        # =========================
        while len(all_landmarks) < 84:
            all_landmarks.append(0)

        key = cv2.waitKey(1)

        if key == ord('s'):

            sample = all_landmarks.copy()

            sample.append(gesture_name)

            data.append(sample)

            print("✅ Sample saved:", len(data))

    cv2.putText(
        frame,
        f"Gesture: {gesture_name}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {len(data)}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("Collect Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)

df.to_csv(
    f"dataset/{gesture_name}.csv",
    index=False
)

print("✅ Dataset saved successfully!")