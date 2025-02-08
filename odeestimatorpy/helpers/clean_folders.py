import json
import os
import shutil

BASE_DIR = "D:\School\Tesis\ode-estimator\output\identifiable"

def clean_folders(directory):
    models_file = os.path.join(directory, "models.json")
    if not os.path.exists(models_file):
        print("models.json not found in the directory.")
        return

    with open(models_file, "r", encoding="utf-8") as f:
        models = json.load(f)

    deleted_indices = []

    for model in models:
        if model is None:
            continue
        model_id = model.get("ID")
        if not model_id:
            continue

        folder_path = os.path.join(directory, model_id)

        if os.path.isdir(folder_path):
            num_elements = len(os.listdir(folder_path))
            if num_elements < 10:
                shutil.rmtree(folder_path)
                index = models.index(model)
                deleted_indices.append(index)
                models[index] = None

    with open(models_file, "w") as f:
        json.dump(models, f, indent=2)

    print("Deleted indices:", deleted_indices)
    return deleted_indices

if __name__ == "__main__":
    clean_folders(BASE_DIR)