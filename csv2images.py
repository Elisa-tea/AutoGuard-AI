from pathlib import Path
import sys
import pandas as pd
from fast_csv2Image_notime import make_can_image

# Paths relative to project root
input_dir = Path("data/can-migru-dataset/csv_format2")
output_base_dir = Path("outputs/imagesGIDS/CAN-MIGRU2")


for csv_file in input_dir.glob("*.csv"):
    base_name = csv_file.stem
    
    # nested output folder
    nested_output_dir = output_base_dir / f"shell_{base_name}" / base_name
    nested_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {csv_file.name}")

    data = pd.read_csv(csv_file)

    make_can_image(data, nested_output_dir)


