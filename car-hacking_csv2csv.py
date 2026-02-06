import pandas as pd
import os

dataset_name = 'RPM'
# Paths relative to project root
base_raw = "data/car-hacking-dataset/raw"
base_out = "data/car-hacking-dataset/csv_format"

input_csv = os.path.join(base_raw, f'{dataset_name}_dataset.csv')
normal_csv = os.path.join(base_out, 'normal', f'normal_{dataset_name}.csv')
attack_csv = os.path.join(base_out, 'attacked', f'attacked_{dataset_name}.csv')
full_db = os.path.join(base_out, 'full_attacked', f'formatted_{dataset_name}.csv')

# data extraction
with open(input_csv, 'r') as f:
    lines = f.readlines()

parsed_rows = []
for line in lines:
    parts = line.strip().split(',')
    try:
        timestamp = float(parts[0])
        timestamp = round(timestamp, 6)
        can_id = parts[1][1:]
        dlc = int(parts[2])
        data_bytes = parts[3:3 + dlc]
        label = parts[3 + dlc]
        data_bytes += [''] * (8 - len(data_bytes))  # pad to 8
        parsed_rows.append([timestamp, can_id, dlc] + data_bytes + [label])
    except Exception as e:
        print(f"[!] Error parsing line: {line}")
        print(f"    -> {e}")
        continue


columns = ['timestamp', 'id', 'dlc'] + [f'D{i}' for i in range(8)] + ['label: R_normal / T_attack']
df = pd.DataFrame(parsed_rows, columns=columns)

df['delta_t'] = df['timestamp'].diff().fillna(0).round(6)
data_cols = [f'D{i}' for i in range(8)]
df['data_bytes'] = df[data_cols].apply(
    lambda row: ' '.join(
        val
        for val in row if pd.notna(val) and str(val).strip() != '' and val is not None
    ), axis=1)


full_df = df[['timestamp', 'delta_t', 'id', 'dlc', 'data_bytes', 'label: R_normal / T_attack']]
normal_df = full_df[full_df['label: R_normal / T_attack'] == 'R'].drop(columns='label: R_normal / T_attack')
attack_df = full_df[full_df['label: R_normal / T_attack'] == 'T'].drop(columns='label: R_normal / T_attack')

os.makedirs(os.path.dirname(normal_csv), exist_ok=True)
os.makedirs(os.path.dirname(attack_csv), exist_ok=True)
os.makedirs(os.path.dirname(full_db), exist_ok=True)

normal_df.to_csv(normal_csv, index=False)
attack_df.to_csv(attack_csv, index=False)
full_df.to_csv(full_db, index=False)

# Sanity checks
print("\nSanity check 1: Missing values")
print(df.isna().sum())
print("Any missing values:", df.isna().any().any())

print("\nSanity check 2: Row counts")
print(f"Total input rows: {len(df)}")
print(f"Normal rows (R): {len(normal_df)}")
print(f"Attack rows (T): {len(attack_df)}")
print(f"Full formatted rows: {len(full_df)}")

print("\nSanity check 3: Previews")
print("\nNormal:")
print(normal_df.head())
print("\nAttack:")
print(attack_df.head())
print("\nFull formatted:")
print(full_df.head())
