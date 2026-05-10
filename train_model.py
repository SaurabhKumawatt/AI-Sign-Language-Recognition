import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATASET_PATHS = [
    "dataset",
    "dataset/custom"
]
MODEL_PATH = "model/gesture_model.pkl"

X = []
y = []

print("🔄 Loading dataset...")

for dataset_path in DATASET_PATHS:

    if not os.path.exists(dataset_path):
        continue

    for file in os.listdir(dataset_path):

        if file.endswith(".csv"):

            label = file.replace(".csv", "")

            file_path = os.path.join(
                dataset_path,
                file
            )

            try:
                df = pd.read_csv(file_path)

            except:
                print(f"⚠️ Skipping invalid file: {file}")
                continue

            for row in df.values:

                try:

                    features = row[:-1]

                    # skip invalid rows
                    if len(features) != 84:
                        continue

                    features = np.array(features, dtype=float)

                    X.append(features)
                    y.append(label)

                except:
                    continue

X = np.array(X, dtype=float)
y = np.array(y)

print(f"✅ Total samples: {len(X)}")
print(f"✅ Total gestures: {len(set(y))}")

# 🔹 Split data (VERY IMPORTANT)
if len(X) == 0:
    print("❌ No valid dataset found")
    exit()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("📊 Training samples:", len(X_train))
print("📊 Testing samples:", len(X_test))


# =========================
# 🔥 OPTION 1: KNN (Simple)
# =========================
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
knn_acc = accuracy_score(y_test, y_pred_knn)

print(f"✅ KNN Accuracy: {knn_acc * 100:.2f}%")



# =========================
# 🔥 OPTION 2: Random Forest (Better)
# =========================
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)

print(f"✅ Random Forest Accuracy: {rf_acc * 100:.2f}%")


# 🔹 Choose best model
if rf_acc > knn_acc:
    model = rf_model
    print("🚀 Using Random Forest")
else:
    model = knn_model
    print("🚀 Using KNN")


# 🔹 Save model
os.makedirs("model", exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("🎉 Model saved at:", MODEL_PATH)