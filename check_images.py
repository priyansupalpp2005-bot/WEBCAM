from PIL import Image
import os

dataset_path = "dataset"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        path = os.path.join(root, file)

        try:
            img = Image.open(path)
            img.verify()
        except:
            print("Corrupted:", path)