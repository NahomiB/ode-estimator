import concurrent.futures
import json
import os
import re
import traceback

import numpy as np

from tqdm import tqdm

from odeestimatorpy.estimators.pso_estimator import PSOEstimator
from odeestimatorpy.helpers.collect_models import load_existing_models
from odeestimatorpy.helpers.save_to_json import save_new_json
from odeestimatorpy.models.linear_ode_model import LinearODEModel

OUTPUT_DIR = r"..\..\output\identifiable"
SMOOTH_DATA_PATTERN = re.compile(r"smoothed_data_(5|10|15)\.json")

def process_single_system(system: dict):
    """Process a single ODE system: integrate, add noise, smooth, and estimate parameters."""

    if "ID" not in system.keys():
        return

    system_dir = os.path.join(OUTPUT_DIR, system["ID"])

    unconstrained_path = os.path.join(system_dir, "unconstrained.json")

    if not os.path.exists(unconstrained_path):
        model_dict = apply_constraints(system)
        model = LinearODEModel.from_dict(model_dict)

        save_new_json(model.export(), unconstrained_path)
    else:
        with open(unconstrained_path, "r") as f:
            model_dict = json.load(f)
        model = LinearODEModel.from_dict(model_dict)


    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process, model, system_dir, file)
            for file in os.listdir(system_dir) if SMOOTH_DATA_PATTERN.match(file)
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Ensure all tasks complete

    return system["ID"]

def process(model: LinearODEModel, system_dir: str, file: str):

    try:
        with open(os.path.join(system_dir, file), "r") as f:
            smoothed_data = json.load(f)["data"]

        with (open(os.path.join(system_dir, "data.json"), "r")) as f:
            x = json.load(f)["x"]

    except FileNotFoundError:
        smoothed_data = None

    if smoothed_data is None or x is None:
        return

    try:

        smoothed_data = np.array(smoothed_data)

        estimator = PSOEstimator(model,  np.column_stack((x, smoothed_data.T)))
        estimated_params = estimator.solve()

        param_file = f"{system_dir}/pso_parameters_{file}"
        save_new_json({"parameters": estimated_params}, param_file)

    except Exception:
        traceback.print_exc()
        print("hi")



def apply_constraints(system_data):
    """
    Apply parameter constraints to update the equations with unified parameter names.

    Args:
        system_data (dict): Dictionary containing equations, variables, parameters, and constraints.

    Returns:
        dict: Updated system data with modified equations.
    """
    # Create a mapping of equivalent parameters
    param_map = {}
    for constraint in system_data["constraints"]:
        param1, param2 = constraint.split(" == ")
        canonical_name = min(param1, param2)  # Choose a consistent name (lexicographically smallest)
        param_map[param1] = canonical_name
        param_map[param2] = canonical_name

    # Update equations with unified parameter names
    updated_equations = []
    for equation in system_data["equations"]:
        for param, unified_name in param_map.items():
            equation = re.sub(rf'\b{param}\b', unified_name, equation)
        updated_equations.append(equation)

    # Return updated system data
    updated_system = system_data.copy()
    updated_system["equations"] = updated_equations
    updated_system["constraints"] = []
    updated_system["parameters_names"] = list(set(param_map.values()))
    updated_system["parameters"] = []
    return updated_system


def process_ode_systems(batch_size=100):
    """Process ODE systems in batches using multiprocessing for parallelism."""

    ode_systems = load_existing_models(OUTPUT_DIR)

    total_systems = len(ode_systems)
    print(f"Total systems to process: {total_systems}")

    for i in tqdm(range(0, total_systems, batch_size), desc="Processing ODE systems"):
        batch = ode_systems[i: i + batch_size]

        # Use multiprocessing to process each ODE system in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(process_single_system, batch))


if __name__ == "__main__":
    process_ode_systems(batch_size=100)
