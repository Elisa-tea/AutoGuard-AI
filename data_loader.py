import os
import glob
import numpy as np
from PIL import Image

def load_images_from_folders(folder_names, label, base_path, image_size=(64, 64)):
    X = []
    y = []
    for folder in folder_names:
        folder_path = os.path.join(base_path, folder)
        image_paths = glob.glob(os.path.join(folder_path, "*.png"))
        for img_path in image_paths:
            img = Image.open(img_path).convert('L').resize(image_size)
            img_array = np.array(img)
            X.append(img_array)
            y.append(label)
    return np.array(X), np.array(y)
