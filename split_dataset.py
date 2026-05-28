import os
import shutil
from sklearn.model_selection import train_test_split

source_dir = "dataset/asl_alphabet_train" 

train_dir = "dataset/train"
test_dir = "dataset/test"

for label in os.listdir(source_dir):

    label_path = os.path.join(source_dir, label)

    if not os.path.isdir(label_path):
        continue

    images = [
    f for f in os.listdir(label_path)
    if os.path.isfile(os.path.join(label_path, f))
]

    train_images, test_images = train_test_split(
        images,
        test_size=0.2,
        random_state=42
    )

    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(test_dir, label), exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(label_path, img),
            os.path.join(train_dir, label, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(label_path, img),
            os.path.join(test_dir, label, img)
        )

print("Dataset split completed!")