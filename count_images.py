import os

dataset_path = "dataset"

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):
        count = len(os.listdir(folder_path))
        print(folder, ":", count)