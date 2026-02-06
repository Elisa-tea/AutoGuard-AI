import csv
fl = 1
def parse_log_line(line):
    try:
      
        parts = line.strip().split()
        timestamp = float(parts[0].strip("()"))
        id_data = parts[2]
        can_id, raw_data = id_data.split("#")
        can_id = can_id.lower()
        data_bytes = " ".join([raw_data[i:i+2] for i in range(0, len(raw_data), 2)])
        dlc = len(raw_data) // 2
        label = 'R' if parts[3] == '0' else 'T'
        
        return timestamp, can_id, dlc, data_bytes, label
    except Exception as e:
        print(f"Failed to parse line: {line}")
        return None

def convert_log_to_csv(log_path, csv_path):
    rows = []
    prev_time = None

    with open(log_path, 'r') as log_file:
        for line in log_file:
            parsed = parse_log_line(line)
            if parsed:
                timestamp, can_id, dlc, data_bytes, label = parsed
                if prev_time is None:
                    delta_t = 0.0
                else:
                    delta_t = round(timestamp - prev_time, 6)
                prev_time = timestamp
                
                rows.append([timestamp, delta_t, can_id, dlc, data_bytes, label])
        

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'delta_t', 'id', 'dlc', 'data_bytes', 'label: R_normal / T_attack'])
        writer.writerows(rows)


# Paths relative to project root
log_file_path = "data/can-migru-dataset/raw/Benign/Day_5/Benign_day5_file1.log"
csv_file_path = "data/can-migru-dataset/csv_format/benign_d5.csv"

convert_log_to_csv(log_file_path, csv_file_path)
