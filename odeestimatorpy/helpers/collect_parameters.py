# Base directory to search for files
import json
import os
from typing import re

BASE_DIR = "D:\School\Tesis\ode-estimator\output\identifiable"

# Regex pattern to match the required filenames
FILENAME_PATTERN = re.compile(r"parameter_estimations_(5|10|15)\.json")

def collect_parameters(base_dir):
    # Counters and storage
    numeric_count = 0
    non_numeric_count = 0
    error_folders = []

    for root, _, files in os.walk(base_dir):
        non_numeric_by_folder = 0
        for file in files:
            if FILENAME_PATTERN.match(file):  # Check if filename matches the expected pattern
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        if 'parameters' in data and any(not isinstance(value, float) for value in data['parameters'].values()):
                            non_numeric_by_folder += 1

                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error reading {file_path}: {e}")

        if non_numeric_by_folder > 0:
            if non_numeric_by_folder != 3:
                error_folders.append(root)
            non_numeric_count += 1
        else:
            numeric_count += 1


    # Print results
    print(f"Numeric values count: {numeric_count}")
    print(f"Non-numeric values count: {non_numeric_count}")
    print(f"Error folders: {len(error_folders)}")

if __name__ == "__main__":
    collect_parameters(BASE_DIR)