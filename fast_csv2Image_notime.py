import argparse
import numpy as np
import pandas as pd
from PIL import Image
import os

def one_hot_vector(c):
    hex_map = {
        "0": 0, "1": 1, "2": 2, "3": 3,
        "4": 4, "5": 5, "6": 6, "7": 7,
        "8": 8, "9": 9, "a": 10, "b": 11,
        "c": 12, "d": 13, "e": 14, "f": 15
    }
    ohv = np.zeros(16, dtype=np.uint8)
    if isinstance(c, str) and c.lower() in hex_map:
        ohv[hex_map[c.lower()]] = 255
    return ohv

def convert_can_row_to_vector(row):
    id_str = str(row["id"]).rjust(3, "0")[-3:]
    id_vector = np.concatenate([one_hot_vector(c) for c in id_str])

    try:
        dlc_val = int(row["dlc"])
    except:
        dlc_val = 0

    data_str = str(row["data_bytes"]).replace(" ", "").lower()
    n_real = dlc_val * 2
    real_data = data_str[:n_real]
    padded_data = real_data.ljust(16, "f")  # pad with 'f'

    data_vector = np.concatenate([one_hot_vector(c) for c in padded_data])
    dlc_pixel = np.full((1,), int(min(dlc_val, 8) / 8 * 255), dtype=np.uint8)

    return np.concatenate([id_vector, data_vector, dlc_pixel])

def make_can_image(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    row_size = 16
    column_size = 16 * 3 + 16 * 16 + 1
    can_img_num = 0

    for w in range(0, len(data), row_size):
        can_image = np.zeros((0, column_size), dtype=np.uint8)
        batch = data.iloc[w: w + row_size]

        is_malicious_window = (batch['label: R_normal / T_attack'] == 'T').any()

        for _, row in batch.iterrows():
            vector = convert_can_row_to_vector(row)
            if vector.shape[0] != column_size:
                raise ValueError(f"Unexpected vector shape {vector.shape[0]} != {column_size}")
            can_image = np.vstack([can_image, vector])

        image = Image.fromarray(can_image)
        label_str = "m" if is_malicious_window else "n"
        image.save(os.path.join(output_dir, f"can_img_{can_img_num}_{label_str}.png"))
        can_img_num += 1
"""
def make_can_image_majority(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    row_size = 16
    column_size = 16 * 3 + 16 * 16 + 1
    can_img_num = 0
    num_malicious = 0
    num_benign = 0

    for w in range(0, len(data), row_size):
        can_image = np.zeros((0, column_size), dtype=np.uint8)
        batch = data.iloc[w: w + row_size]

        if len(batch) < row_size:
            continue

        malicious_count = (batch['label: R_normal / T_attack'] == 'T').sum()
        is_malicious_window = malicious_count > (row_size // 2)

        for _, row in batch.iterrows():
            vector = convert_can_row_to_vector(row)
            can_image = np.vstack([can_image, vector])

        image = Image.fromarray(can_image)
        label_str = "m" if is_malicious_window else "n"
        image.save(os.path.join(output_dir, f"can_img_{can_img_num}_{label_str}.png"))
        can_img_num += 1

        if is_malicious_window:
            num_malicious += 1
        else:
            num_benign += 1

    print("Majority version complete. Total images created:", can_img_num)
    print(f"Malicious: {num_malicious}, Benign: {num_benign}")
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CAN CSV data into image format")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument("output_dir", help="Directory to save generated images")
    parser.add_argument("--majority", action="store_true", help="Use majority labeling for image labels")

    args = parser.parse_args()

    data = pd.read_csv(args.csv_path)
    print("Total rows loaded:", len(data))
    print("Example row:\n", data.head(1).to_string())
    """
    if args.majority:
        make_can_image_majority(data, args.output_dir)
    else:
        make_can_image(data, args.output_dir)
    """
        
    make_can_image(data, args.output_dir)
