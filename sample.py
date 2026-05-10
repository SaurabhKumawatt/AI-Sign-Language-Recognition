import os
import pandas as pd

dataset_path = "dataset"

print("Gesture-wise sample count:\n")

total = 0

for file in os.listdir(dataset_path):
    if file.endswith(".csv"):
        gesture = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(dataset_path, file))
        count = len(df)
        total += count
        print(f"{gesture} : {count}")

print("\nTotal Samples:", total)