import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

gesture_name = input("Enter gesture label: ")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

data = []

print("Press 's' to save sample")
print("Press 'q' to quit")

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            # ✅ Wrist reference (landmark 0)
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y

            # ✅ Relative coordinates (ONLY x, y)
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x - base_x)
                landmarks.append(lm.y - base_y)

            # ✅ Convert to numpy for normalization
            landmarks = np.array(landmarks)

            # ✅ Scale normalization
            max_value = np.max(np.abs(landmarks))
            if max_value != 0:
                landmarks = landmarks / max_value

            key = cv2.waitKey(1)

            if key == ord('s'):
                sample = landmarks.tolist()
                sample.append(gesture_name)
                data.append(sample)

                print("Sample saved:", len(data))

    cv2.putText(frame, f"Gesture: {gesture_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Collect Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df.to_csv(f"dataset/{gesture_name}.csv", index=False)

print("✅ Dataset saved!")