import json
import os
import concurrent.futures

import numpy as np
from tqdm import tqdm

from odeestimatorpy.estimators.kkt_estimator import KKTLinearODEParameterEstimator
from odeestimatorpy.helpers.collect_models import load_existing_models
from odeestimatorpy.helpers.save_to_json import save_new_json
from odeestimatorpy.models.linear_ode_model import LinearODEModel

def process(save_dir, x, noise_level, model, y=None):

    if y is None:

        try:
            with open(os.path.join(save_dir, f"smoothed_data_{int(noise_level*100)}.json"), "r") as f:
                data = json.load(f)["data"]

        except FileNotFoundError:
            return

        y = np.asarray(data).T

    estimator = KKTLinearODEParameterEstimator(model, np.column_stack((x, y)))
    estimated_params = estimator.solve()
    param_file = f"{save_dir}/kkt_parameter_estimations_{int(noise_level * 100)}.json"
    save_new_json({"parameters": estimated_params}, param_file)

def process_single_system(system):

    save_dir = system['directory']
    model = LinearODEModel.from_dict(system)

    data_path = os.path.join(save_dir, "data.json")

    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            data = json.load(f)
            x = np.array(data["x"])
            y = np.array(data["y"]).T
    else:
       return

    # Process different noise levels in parallel
    noise_levels = [0.05, 0.10, 0.15]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process, save_dir, x, noise_level, model, None)
            for noise_level in noise_levels
        }

        noise_free_future = executor.submit(process, save_dir, x, 0, model, y)
        futures.add(noise_free_future)

        for future in concurrent.futures.as_completed(futures):
            future.result()  # Ensure all tasks complete


def process_ode_systems(batch_size=100):
    """Process ODE systems in batches using multiprocessing for parallelism."""

    ode_systems = load_existing_models("../../outputs-one")
    total_systems = len(ode_systems)
    print(f"Total systems to process: {total_systems}")

    for i in tqdm(range(0, total_systems, batch_size), desc="Processing ODE systems"):
        batch = ode_systems[i: i + batch_size]

        # Use multiprocessing to process each ODE system in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(process_single_system, batch))

if __name__ == "__main__":
    process_ode_systems()