import cv2
import mediapipe as mp
import os
import csv
import numpy as np

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# =========================
# DATASET PATH
# =========================
dataset_path = "dataset/train"

# =========================
# CSV FILE
# =========================
csv_file = open("landmarks.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)

# Header
header = []

for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]

header.append("label")

csv_writer.writerow(header)

# =========================
# PROCESS IMAGES
# =========================
for label in os.listdir(dataset_path):

    label_path = os.path.join(dataset_path, label)

    if not os.path.isdir(label_path):
        continue

    print(f"Processing {label}...")

    for image_name in os.listdir(label_path):

        image_path = os.path.join(label_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                landmarks = []

                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)

                # =========================
                # NORMALIZATION
                # =========================
                landmarks = np.array(landmarks)

                # wrist-relative normalization
                landmarks = landmarks - landmarks[0]

                row = landmarks.tolist()

                row.append(label)

                csv_writer.writerow(row)

csv_file.close()

print("Landmarks extraction completed.")