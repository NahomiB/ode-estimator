import os
import json
import shutil# This is a sample Python script.

import numpy as np

from odeestimatorpy.estimators.kkt_estimator import KKTLinearODEParameterEstimator
from odeestimatorpy.models.linear_ode_model import LinearODEModel


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    directory_path = "D:\School\Tesis\ode-estimator\output\identifiable"
    clean_folders(directory_path)
    print("finished")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
