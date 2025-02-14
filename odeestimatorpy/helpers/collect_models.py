import json
import os
from glob import glob
import re


def load_existing_models(output_dir):
    ode_systems = []
    pattern = re.compile(r"models_(\d+)\.json")

    for file in sorted(glob(os.path.join(output_dir, "**", "models_*.json"), recursive=True)):
        match = pattern.search(file)
        if match:
            with open(file, "r") as f:
                model_list = json.load(f)
                for model in model_list:
                    model["directory"] = os.path.join(os.path.dirname(file), model["ID"])
                ode_systems.extend(model_list)

    return ode_systems


def load_models_in_batches(path_dict, batch_size=100):
    # Load all models from the JSON files
    all_models = {name: load_json_list(path) for name, path in path_dict.items()}

    # Initialize index counters for each model
    indices = {name: 0 for name in all_models.keys()}

    while any(indices):
        batch = {}

        for name in list(indices.keys()):  # Iterate over a copy of the keys
            start = indices[name]
            end = min(start + batch_size, len(all_models[name]))

            if start < len(all_models[name]):  # Only add if there are remaining elements
                batch[name] = all_models[name][start:end]
                indices[name] = end  # Update the index

            # Remove from indices if the model is fully processed
            if indices[name] >= len(all_models[name]):
                del indices[name]

        if batch:  # Only yield non-empty batches
            yield batch


def load_json_list(path):
    """ Loads a JSON file containing a list of models """
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)  # Assumes the JSON contains a list
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {path}: {e}")
        return []


