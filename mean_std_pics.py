import os
import numpy as np
from PIL import Image

# Path relative to project root
image_folder = "outputs/imagesGIDS/shell_n/normal"
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")]

all_pixels = []

for file in image_files:
    try:
        img = Image.open(file).convert("L")  # grayscale
        img_array = np.array(img).astype(np.float32)
        all_pixels.append(img_array.flatten())
    except Exception as e:
        print(f"Failed to read {file}: {e}")

if all_pixels:
    all_pixels_combined = np.concatenate(all_pixels)
    mean_pixel = np.mean(all_pixels_combined)
    std_pixel = np.std(all_pixels_combined)
    print(f"Mean pixel value: {mean_pixel:.4f}")
    print(f"Std pixel value: {std_pixel:.4f}")
else:
    print("No image data found.")
