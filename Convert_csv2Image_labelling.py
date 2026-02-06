import numpy as np
import pandas as pd
from PIL import Image
import os

# Paths relative to project root
csv_path = "data/can-migru-dataset/csv_format/benign_d5.csv"
output_dir = "outputs/imagesGIDS/CAN-MIGRU/shell_nd5/nd5"


os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


 
def one_hot_vector(c):
    """
    def one_hot_vector(c): one hot encoder for a hex character
    INPUT: c - hex character
    OUTPUT: ohv - vector of shape [1, 16], eg [0 0 255 0 .. 0]
    """
    hex_map = {
        "0": 0, "1": 1, "2": 2, "3": 3,
        "4": 4, "5": 5, "6": 6, "7": 7,
        "8": 8, "9": 9, "a": 10, "b": 11,
        "c": 12, "d": 13, "e": 14, "f": 15
    }
    ohv = np.zeros(16, dtype=np.uint8) # vector of shape [1, 16], eg [0 0 255 0 .. 0]
    if isinstance(c, str) and c.lower() in hex_map:
        ohv[hex_map[c.lower()]] = 255
    return ohv


def convert_can_row_to_vector(row):
    """
    def convert_can_row_to_vector(row): converts extracted daatbase row into a row of concat one hot values of id, data, delta, dlc
    INPUT: database row
    OUTPUT: concat of one hot values
    """
    # ID
    try:
        id_str = str(row["id"])
        if len(id_str) != 3:
            raise ValueError(f"Error: something wrong with the length of ID. The length of ID must be 3. The length of current ID {id_str} is {len(id_str)}")
    except:
        raise ValueError(f"Invalid ID: {row['id']}")
    id_vector = np.concatenate([one_hot_vector(c) for c in id_str])


    # Data field
    data_str = str(row["data_bytes"]).replace(" ", "").lower()
    data_str = data_str.ljust(16, "0")[:16]
    data_vector = np.concatenate([one_hot_vector(c) for c in data_str])

    # Delta time
    try:
        delta = float(row["delta_t"])
    except:
        print(f"Be aware! Delta time conversion has just failed for ID:{id_str}")
        delta = 0.0
    delta_corrected = min(max(delta, 0), 1.0)
    # check that delta is valid: careful here with underflow / overflow
    delta_pixel = np.full((1,), int(delta_corrected * 255), dtype=np.uint8)
  
    # DLC
    try:
        dlc_val = int(row["dlc"])
        # first one is safer
        # dlc_pixel = np.full((1,), int(min(max(dlc_val, 0), 8) / 8 * 255), dtype=np.uint8)
        dlc_pixel = np.full((1,), int(dlc_val / 8 * 255), dtype=np.uint8)
    except:
        print(f"Be aware! DLC conversion failed for ID: {id_str}")
        dlc_pixel = np.zeros((1,), dtype=np.uint8)

    return np.concatenate([id_vector, data_vector, delta_pixel, dlc_pixel])



def make_can_image(data):
    """
    def make_can_image(data): CAN time series images generator
    INPUT: data - database
    OUTPUT: latent output - creating and saving images
    """

    row_size = 64 # window size
    column_size = 16 * 3 + 16 * 16 + 1 + 1  # 306: ID (hex; 3-can) + data(hex, 8 bytes) + delta + dlc
    can_img_num = 0 # sanity check

    for w in range(0, len(data), row_size):
        can_image = np.zeros((0, column_size), dtype=np.uint8)
        batch = data.iloc[w : w + row_size]
        
        # remember the label, logic: if any row in the winodw is malicious -> window is malicious
        is_malicious_window = (batch['label: R_normal / T_attack'] == 'T').any()

        for _, row in batch.iterrows():
            vector = convert_can_row_to_vector(row)
            can_image = np.vstack([can_image, vector])

        image = Image.fromarray(can_image)
        label_str = "m" if is_malicious_window else "n"                                                         
        image.save(os.path.join(output_dir, f"can_img_{can_img_num}_{label_str}.png"))
        can_img_num += 1

    print("CAN image generation complete. Total images created:", can_img_num)



def make_can_image_majority(data, output_dir):
    row_size = 64
    column_size = 16 * 3 + 16 * 16 + 1 + 1
    can_img_num = 0

    num_malicious = 0
    num_benign = 0

    for w in range(0, len(data), row_size):
        can_image = np.zeros((0, column_size), dtype=np.uint8)
        batch = data.iloc[w: w + row_size]

        if len(batch) < row_size:
            continue  # skip incomplete window

        malicious_count = (batch['label: R_normal / T_attack'] == 'T').sum()
        is_malicious_window = malicious_count > (row_size // 2)

        for _, row in batch.iterrows():
            vector = convert_can_row_to_vector(row)
            if vector.shape[0] != column_size:
                raise ValueError(f"Unexpected vector shape {vector.shape[0]} != {column_size}")
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


if __name__ == "__main__":

    data = pd.read_csv(csv_path)
    print("Total rows loaded:", len(data))

  
    data.dropna(subset=["id", "data_bytes", "timestamp", "delta_t"], inplace=True)
    print("Rows after dropna:", len(data))

    print("Example row:\n", data.head(1).to_string())

    make_can_image(data)
