import json
import os
import shutil
from glob import glob


def load_existing_models(output_dir):

    models = []
    for file in sorted(glob(os.path.join(output_dir, "models_*.json"))):
        with open(file, "r") as f:
            models.extend(json.load(f))

    return models

SOURCE_DIR = "D:\School\Tesis\ode-estimator\output\identifiable"
OUTPUT_DIR = "..\..\output\identifiable-by-page"

def copy_to_page_folder():

    for file in sorted(glob(os.path.join(SOURCE_DIR, "models_*.json"))):
        with open(file, "r") as f:
            data = json.load(f)
            ids = {str(item["ID"]) for item in data}

        number = file.split("_")[1].split(".")[0]

        number_folder = os.path.join(OUTPUT_DIR, number)
        os.makedirs(number_folder, exist_ok=True)

        for folder_name in os.listdir(SOURCE_DIR):
            folder_path = os.path.join(SOURCE_DIR, folder_name)
            if os.path.isdir(folder_path) and folder_name in ids:
                destination_folder = os.path.join(number_folder, folder_name)
                shutil.copytree(folder_path, destination_folder, dirs_exist_ok=True)

    print(os.path.join(SOURCE_DIR, "models_*.json"))

    print(len(glob(os.path.join(SOURCE_DIR, "models_*.json"))))