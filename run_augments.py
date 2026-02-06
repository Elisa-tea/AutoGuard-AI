import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'preprocessing')))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_cleaning import clean_text_file, parse_cleaned_file
from augment_traditional import upsample, add_noise, reverse_data, random_drift
from augment_vae import train_vae, generate_vae_samples
from correct_timestamps import regenerate_timestamps_similar

# Paths relative to project root.
# Update locally if dataset is stored elsewhere.
input_path = "data/car-hacking-dataset/raw/normal_run_data.txt"
clean_path = "data/car-hacking-dataset/preprocessed/cleaned_normal_run_data.txt"

def to_byte(v):
    if isinstance(v, str):
        try:
            return int(v, 16)
        except ValueError:
            return int(float(v) * 255)
    else:
        return int(float(v) * 255)
        
def convert_byte(row, idx):
    # hex string to int
    if idx < row['DLC']:
        val = row.get(f'D{idx}', '00')
        if pd.isna(val):
            return np.nan
        try:
            return int(str(val), 16)
        except ValueError:
            return np.nan
    else:
        return np.nan


def format_row(row):
    try:
        data_columns = ['D0','D1','D2','D3','D4','D5','D6','D7']
        data_bytes = [to_byte(row[col]) for col in data_columns]
        return (
            f"Timestamp: {float(row['Timestamp']):.6f}\t"
            f"ID: {row['ID']}\t000\t"
            f"DLC: {int(row['DLC'])}\t"
            f"Data: {' '.join(f'{b:02X}' for b in data_bytes)}"
        )
    except Exception as e:
        print(f"[ERROR] Failed to format row:\n{row}")
        raise e


def save_df_to_txt(df, filename):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            f.write(format_row(row) + "\n")


clean_text_file(input_path, clean_path)
df = parse_cleaned_file(clean_path)
df.to_csv('benign_traffic.csv', index=False)

df['DLC'] = df['DLC'].astype(int)

payload = df.apply(lambda row: [convert_byte(row, i) for i in range(8)], axis=1, result_type='expand')
payload.columns = [f'D{i}' for i in range(8)]

# missing values
for col in payload.columns:
    valid_vals = payload.loc[payload[col].notna(), col]
    mean_val = valid_vals.mean()
    payload[col] = payload[col].fillna(mean_val)


payload = payload.round().astype(int)

scaler = MinMaxScaler()
data_array = scaler.fit_transform(payload.values.astype('float32'))

USE_VAE = True
USE_TRAD = False

vae_df = pd.DataFrame()
trad_df = pd.DataFrame()
n_total = 2 * len(df)

if USE_VAE:
    vae = train_vae(data_array, epochs=5, latent_dim=4, batch_size=64)
    vae_df = generate_vae_samples(vae, scaler, int(n_total * 0.3), latent_dim=4)

    # VAE output to hex strings
    for c in [f'D{i}' for i in range(8)]:
        vae_df[c] = vae_df[c].clip(0, 255).round().astype(int).apply(lambda x: format(x, '02X'))

    # regenerate realistic timestamps
    vae_df['Timestamp'] = regenerate_timestamps_similar(df['Timestamp'].astype(float).values, len(vae_df))
    vae_df['ID'] = df['ID'].sample(len(vae_df), replace=True).values

    # 8-byte CAN frames
    vae_df['DLC'] = 8


    save_df_to_txt(vae_df, "data/car-hacking-dataset/raw/val2.txt")


if USE_TRAD:
    # upsample
    base_df = upsample(df, factor=(n_total * 0.7) / len(df))

    # add_noise to all rows
    noisy_df = add_noise(base_df.copy())

    # reverse to 10% of the noisy data
    n_reverse = int(0.1 * len(base_df))
    reversed_df = reverse_data(base_df.sample(n=n_reverse, random_state=42).copy())

    # drift 10% of the noisy data
    n_drift = int(0.1 * len(base_df))
    drifted_df = random_drift(base_df.sample(n=n_drift, random_state=43).copy())

    trad_df = pd.concat([noisy_df, reversed_df, drifted_df], ignore_index=True)

    # float conversion
    new_timestamps = regenerate_timestamps_similar(df['Timestamp'].astype(float).values, len(trad_df))
    trad_df['Timestamp'] = new_timestamps

    save_df_to_txt(trad_df, 'trad.txt')

if USE_VAE and USE_TRAD:
    augmented = pd.concat([vae_df, trad_df], ignore_index=True)
    augmented.to_csv('augmented_normal_run.csv', index=False)
