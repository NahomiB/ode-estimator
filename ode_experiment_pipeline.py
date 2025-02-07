import json
import os
import uuid
import concurrent.futures
import numpy as np
from tqdm import tqdm

from odeestimatorpy.data_generator.noise.noise_adder import NoiseAdder
from odeestimatorpy.data_generator.ode_integrator import ODEIntegrator
from odeestimatorpy.data_generator.spline_cross_validator import SplineCrossValidator
from odeestimatorpy.data_generator.spline_smoother import SplineSmoother
from odeestimatorpy.estimators.kkt_estimator import KKTLinearODEParameterEstimator
from odeestimatorpy.models.linear_ode_model import LinearODEModel

input_file = "examples/odes_identifiable.json"
output_dir = "output/identifiable/"


def process_single_system(system):
    """Process a single ODE system: integrate, add noise, smooth, and estimate parameters."""
    system_id = str(uuid.uuid4())
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
    save_json(model_dict, f"{output_dir}models.json")

    # Generate data from the model
    data = ODEIntegrator(model, [0, 100]).integrate(num_points=500)
    x = data["x"]
    y = data["y"]

    save_new_json({"x": x.tolist(), "y": y.tolist()}, f"{system_dir}/data.json")

    # Process different noise levels in parallel
    noise_levels = [0.05, 0.10, 0.15]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_noise_level, system_dir, x, y, noise_level, model)
            for noise_level in noise_levels
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Ensure all tasks complete

    return system_id


def process_noise_level(system_dir, x, y, noise_level, model):
    """Add noise, smooth data with the best 's' value, and estimate parameters."""

    noise_adder = NoiseAdder(noise_type="proportional", noise_level=noise_level)
    noisy_data = noise_adder.add_noise(y)
    noise_file = f"{system_dir}/data_with_noise_{int(noise_level * 100)}.json"
    save_new_json({"data": noisy_data.tolist()}, noise_file)

    # Find the best 's' using cross-validation
    best_s = find_best_s(x, noisy_data)

    # Smooth the noisy data using the best 's'
    smoother = SplineSmoother(s_value=best_s)
    smoothed_data = smoother.smooth(x, noisy_data)
    smooth_file = f"{system_dir}/smoothed_data_{int(noise_level * 100)}.json"
    save_new_json({"data": smoothed_data.tolist(), "s": best_s}, smooth_file)

    # Estimate parameters
    estimator = KKTLinearODEParameterEstimator(model, np.column_stack((x, smoothed_data.T)))
    estimated_params = estimator.solve()
    param_file = f"{system_dir}/parameter_estimations_{int(noise_level * 100)}.json"
    save_new_json({"parameters": estimated_params}, param_file)


def find_best_s(x, y, s_values=None):
    """Performs cross-validation to determine the best 's' value for smoothing."""
    if s_values is None:
        s_values = np.logspace(-2, 2, 30)  # Example range of s values

    validator = SplineCrossValidator(x, y)
    means = validator.cross_validate(s_values)
    s, mean = SplineCrossValidator.get_best_s(means)

    return s

def validate_s(s, x, y, kf):
    """Compute validation error for a given s."""
    errors = []

    for train_idx, test_idx in kf.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[:, train_idx], y[:, test_idx]

        smoother = SplineSmoother(s_value=s)
        y_pred = smoother.smooth(x_train, y_train)

        error = np.mean((y_pred - y_test) ** 2)  # Mean Squared Error
        errors.append(error)

    return s, np.mean(errors)


def process_ode_systems(batch_size=100):
    """Process ODE systems in batches using multiprocessing for parallelism."""
    print("Starting to process ODE systems...")

    with open(input_file, "r") as f:
        ode_systems = json.load(f)

    total_systems = len(ode_systems)
    print(f"Total systems to process: {total_systems}")

    for i in tqdm(range(0, total_systems, batch_size), desc="Processing ODE systems"):
        batch = ode_systems[i: i + batch_size]

        # Use multiprocessing to process each ODE system in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(process_single_system, batch))


def save_json(data, file_path):
    """Save JSON data to a file, appending if necessary."""
    try:
        with open(file_path, "r") as f:
            existing_data = json.load(f)
    except Exception:
        existing_data = []

    existing_data.append(data)
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=2)


def save_new_json(data, file_path):
    """Save JSON data to a new file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    process_ode_systems(batch_size=100)