import csv
import re
import pandas as pd

# Paths relative to project root
input_file = "data/dataset/raw/Soul/FreeDrivingData_20180112_KIA.txt"
output_file = "data/dataset/csv_format/Soul/normal.csv"

pattern = re.compile(
    r"Timestamp:\s+([\d.]+)\s+ID:\s+([0-9a-fA-F]+)\s+\d+\s+DLC:\s+(\d+)\s+((?:[0-9a-fA-F]{2}\s*)+)"
)

last_timestamp = None
row_count = 0  # sanity check variable
total_lines = 0 # sanity check variable
preview_limit = 5 # visual sanity check variable
preview_rows = [] # visual sanity check variable

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['timestamp', 'delta_t', 'id', 'dlc', 'data_bytes']) 

    for line in infile:
        total_lines += 1
        match = pattern.search(line)
        if match:
            # data extraction
            timestamp = float(match.group(1))
            can_id = match.group(2)[1:]
            dlc = int(match.group(3))
            data_str = match.group(4).strip()
            data_bytes = ' '.join(data_str.split())

            # get time deltas; initial = 0
            delta_t = 0 if last_timestamp is None else round(timestamp - last_timestamp, 5)
            last_timestamp = timestamp

            # construxt and write a row for the .csv 
            row = [timestamp, delta_t, f'{can_id.lower()}', dlc, data_bytes]
            writer.writerow(row)

            # visual sanity checks
            if row_count < preview_limit:
                preview_rows.append(row)

            row_count += 1



# Sanity checks 1
print()
print("Sanity check 1 >>>")

print()
print(f"Total lines in text file: {total_lines}")
print(f"Valid CAN frames written to CSV: {row_count}")

print()
if row_count == 0:
    print("No valid CAN frames found. Please check the log format.")
else:
    print("First few parsed rows:")
    for r in preview_rows:
        print(r)


# Sanity checks 2
df = pd.read_csv(output_file)
print()
print()
print("Sanity check 2 (after reading with pandas) >>>")
print()

print(f"Number of rows (excluding header): {len(df)}")
counts = df.isna().sum() # NaN/None count sanity check
for col, count in counts.items():
    print(f"{col}: {count}")

print("Missing values exist? ",df.isna().any().any()) # missing values sanity check

# Sanity checks 3 - general info
print()
print()
print("Sanity check 3 >>>")
print(df.info()) 

