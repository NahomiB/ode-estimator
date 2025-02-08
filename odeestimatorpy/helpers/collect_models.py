import json
import os
from glob import glob


def load_existing_models(output_dir):

    models = []
    for file in sorted(glob(os.path.join(output_dir, "models_*.json"))):
        with open(file, "r") as f:
            models.extend(json.load(f))

    return models