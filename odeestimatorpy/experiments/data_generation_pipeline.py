import concurrent.futures
import json
import os
import uuid

import numpy as np

from odeestimatorpy.data_generator.noise.noise_adder import NoiseAdder
from odeestimatorpy.data_generator.ode_integrator import ODEIntegrator
from odeestimatorpy.data_generator.spline_cross_validator import SplineCrossValidator
from odeestimatorpy.data_generator.spline_smoother import SplineSmoother
from odeestimatorpy.helpers.collect_models import load_models_in_batches
from odeestimatorpy.helpers.save_to_json import save_new_json, save_json
from odeestimatorpy.models.linear_ode_model import LinearODEModel


def get_output_path(name, page):
    return f"../../output/{name}/{page}/"

def process_single_system(system):
    """Process a single ODE system: integrate, add noise, smooth, and estimate parameters."""

    save_dir = system['directory']
    model = LinearODEModel.from_dict(system)

    data_path = os.path.join(save_dir, "data.json")

    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            data = json.load(f)
            x = np.array(data["x"])
            y = np.array(data["y"])
    else:
        try:
            # Generate data from the model
            data = ODEIntegrator(model, [0, 100]).integrate(num_points=200)
            x = data["x"]
            y = data["y"]

            save_new_json({"x": x.tolist(), "y": y.tolist()}, data_path)

        except Exception:
            return

    # Process different noise levels in parallel
    noise_levels = [0.05, 0.10, 0.15]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_noise_level, save_dir, x, y, noise_level, model)
            for noise_level in noise_levels
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Ensure all tasks complete


def process_noise_level(save_dir, x, y, noise_level, model):
    """Add noise, smooth data with the best 's' value, and estimate parameters."""

    noise_adder = NoiseAdder(noise_type="additive", noise_level=noise_level)
    noisy_data = noise_adder.add_noise(y)
    noise_file = f"{save_dir}/data_with_noise_{int(noise_level * 100)}.json"
    save_new_json({"data": noisy_data.tolist()}, noise_file)

    # Find the best 's' using cross-validation
    best_s = find_best_s(x, noisy_data)

    # Smooth the noisy data using the best 's'
    smoother = SplineSmoother(s_value=best_s)
    smoothed_data = smoother.smooth(x, noisy_data)
    smooth_file = f"{save_dir}/smoothed_data_{int(noise_level * 100)}.json"
    save_new_json({"data": smoothed_data.tolist(), "s": best_s}, smooth_file)

def find_best_s(x, y, s_values=None):
    """Performs cross-validation to determine the best 's' value for smoothing."""
    if s_values is None:
        s_values = np.logspace(-2, 2, 30)  # Example range of s values

    validator = SplineCrossValidator(x, y)
    means = validator.cross_validate(s_values)
    s, mean = SplineCrossValidator.get_best_s(means)

    return s

def define_system(page, system, output_dir):

    if "ID" not in system:
        system_id = str(uuid.uuid4())
    else:
        system_id = system["ID"]

    system_dir = os.path.join(output_dir, system_id)
    os.makedirs(system_dir, exist_ok=True)

    # Instantiate the model
    model = LinearODEModel.from_dict(system)
    model.set_generated_inputs()
    model.set_generated_parameters()
    model.set_generated_initial_conditions()

    # Generate and save model metadata
    model_dict = model.export()
    model_dict["ID"] = system_id

    save_json(model_dict, f"{output_dir}models_{page}.json")

    model_dict["directory"] = system_dir
    return model_dict

def process_ode_systems(path_dict, batch_size=100):
    batches = load_models_in_batches(path_dict, batch_size=batch_size)
    count = 0

    for batch in batches:

        for name, items in batch.items():

            output = get_output_path(name, page=count)
            items = [define_system(count, item, output) for item in items]

            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(process_single_system, items))

            print(f'Processed page: {count} of {name} with {len(items)} items')

        count += 1

if __name__ == '__main__':

    paths = {
        #"identifiable": r"..\..\examples\odes_identifiable.json",
        #"globally-identifiable": r"..\..\examples\odes_globally_identifiable.json"#,
        "defined-models": r"..\..\examples\odes_defined_models.json"
    }

    process_ode_systems(paths)