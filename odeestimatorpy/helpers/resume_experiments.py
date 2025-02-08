import concurrent.futures

import os
import json
import concurrent.futures
import numpy as np

from odeestimatorpy.data_generator.noise.noise_adder import NoiseAdder
from odeestimatorpy.data_generator.ode_integrator import ODEIntegrator
from odeestimatorpy.data_generator.spline_cross_validator import SplineCrossValidator
from odeestimatorpy.data_generator.spline_smoother import SplineSmoother
from odeestimatorpy.estimators.kkt_estimator import KKTLinearODEParameterEstimator
from odeestimatorpy.experiments.kkt_experiment_pipeline import define_system
from odeestimatorpy.helpers.json import save_new_json
from odeestimatorpy.models.linear_ode_model import LinearODEModel

input_file = "../../examples/odes_identifiable.json"
output_dir = "../../output/identifiable/"
models_file = f"{output_dir}models.json"

required_files = [
    "data.json",
    "data_with_noise_5.json",
    "data_with_noise_10.json",
    "data_with_noise_15.json",
    "smoothed_data_5.json",
    "smoothed_data_10.json",
    "smoothed_data_15.json",
    "parameter_estimations_5.json",
    "parameter_estimations_10.json",
    "parameter_estimations_15.json",
]


def resume_experiment(system_dir, system, missing_steps):
    """Resume an unfinished experiment by checking missing files and re-running necessary steps."""

    if not missing_steps:
        return

    print(f"Resuming experiment in {system_dir}, missing: {missing_steps}")

    model = LinearODEModel.from_dict(system)
    model.set_generated_inputs()
    model.set_generated_parameters()
    model.set_generated_initial_conditions()

    if "data.json" in missing_steps:
        data = ODEIntegrator(model, [0, 100]).integrate(num_points=100)
        save_new_json({"x": data["x"].tolist(), "y": data["y"].tolist()}, f"{system_dir}/data.json")

    with open(f"{system_dir}/data.json", "r") as f:
        data = json.load(f)
        x, y = np.array(data["x"]), np.array(data["y"])

    noise_levels = [0.05, 0.10, 0.15]
    for noise_level in noise_levels:
        noise_file = f"data_with_noise_{int(noise_level * 100)}.json"
        if noise_file in missing_steps:
            noise_adder = NoiseAdder(noise_type="proportional", noise_level=noise_level)
            noisy_data = noise_adder.add_noise(y)
            save_new_json({"data": noisy_data.tolist()}, f"{system_dir}/{noise_file}")


        smooth_file = f"smoothed_data_{int(noise_level * 100)}.json"
        param_file = f"parameter_estimations_{int(noise_level * 100)}.json"

        if smooth_file in missing_steps:
            with open(f"{system_dir}/{noise_file}", "r") as f:
                noisy_data = np.array(json.load(f)["data"])

            best_s = find_best_s(x, noisy_data)
            smoother = SplineSmoother(s_value=best_s)
            smoothed_data = smoother.smooth(x, noisy_data)
            save_new_json({"data": smoothed_data.tolist(), "s": best_s}, f"{system_dir}/{smooth_file}")


        if param_file in missing_steps:

            with open(f"{system_dir}/{smooth_file}", "r") as f:
                smoothed_data = np.array(json.load(f)["data"])

            estimator = KKTLinearODEParameterEstimator(model, np.column_stack((x, smoothed_data.T)))
            estimated_params = estimator.solve()
            save_new_json({"parameters": estimated_params}, f"{system_dir}/{param_file}")


def find_best_s(x, y):
    """Find the best 's' value for smoothing."""
    s_values = np.logspace(-2, 2, 30)
    validator = SplineCrossValidator(x, y)
    means = validator.cross_validate(s_values)
    s, _ = SplineCrossValidator.get_best_s(means)
    return s


def load_existing_models():
    """Load models.json and return a dictionary mapping model details to their assigned ID."""
    try:
        with open(models_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def match_model_dict(one, another):

    one_equations = one["equations"]
    another_equations = another["equations"]

    if len(one_equations) != len(another_equations):
        return False

    for i in range(len(one_equations)):
        if one_equations[i] != another_equations[i]:
            return False

    one_constraints = one["constraints"]
    another_constraints = another["constraints"]

    if len(one_constraints) != len(another_constraints):
        return False

    for i in range(len(one_constraints)):
        if one_constraints[i] != another_constraints[i]:
            return False

    return True

def single(model, existing_models):
    result = [existing_model for existing_model in existing_models if match_model_dict(model, existing_model)]
    return result[0] if result else None

def resume_unfinished_experiments():
    """Find and resume unfinished experiments."""
    with open(input_file, "r") as f:
        ode_systems = json.load(f)

    existing_models = load_existing_models()
    count = len(existing_models)

    unfinished_experiments = []

    for system in ode_systems:

        existing_model = single(system, existing_models)

        if existing_model is None:

            model, system_dir = define_system(count, system)
            count += 1

            unfinished_experiments.append((system_dir, model, required_files))

        system_dir = os.path.join(output_dir, existing_model["ID"])

        if os.path.isdir(system_dir):
            existing_files = set(os.listdir(system_dir))
            missing_steps = [f for f in required_files if f not in existing_files]
            if missing_steps:
                unfinished_experiments.append((system_dir, existing_model, missing_steps))

    if not unfinished_experiments:
        print("No unfinished experiments found.")
        return

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(lambda args: resume_experiment(*args), unfinished_experiments)


if __name__ == "__main__":
    resume_unfinished_experiments()