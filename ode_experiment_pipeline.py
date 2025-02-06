import json
import os
import uuid

import numpy as np
from tqdm import tqdm

from odeestimatorpy.data_generator.noise.noise_adder import NoiseAdder
from odeestimatorpy.data_generator.ode_integrator import ODEIntegrator
from odeestimatorpy.data_generator.spline_smoother import SplineSmoother
from odeestimatorpy.estimators.kkt_estimator import KKTLinearODEParameterEstimator
from odeestimatorpy.models.linear_ode_model import LinearODEModel

input_file = "examples/odes_identifiable.json"
output_dir = "output/identifiable/"


def process_ode_systems(batch_size=100):

    print("Starting to process ODE systems...")

    with open(input_file, "r") as f:
        ode_systems = json.load(f)

    total_systems = len(ode_systems)
    print(f"Total systems to process: {total_systems}")

    for i in tqdm(range(0, total_systems, batch_size), desc="Processing ODE systems"):
        batch = ode_systems[i : i + batch_size]

        for system in batch:
            system_id = str(uuid.uuid4())

            # Instantiate the model
            model = LinearODEModel.from_dict(system)

            model.set_generated_inputs()
            model.set_generated_parameters()
            model.set_generated_initial_conditions()

            # Export the model with the unique ID
            model_dict = model.export()
            model_dict["ID"] = system_id
            save_json(model_dict, f"{output_dir}models.json")

            system_dir = os.path.join(output_dir, system_id)
            os.makedirs(system_dir, exist_ok=True)

            # Generate data from the model
            data = ODEIntegrator(model, [0, 100]).integrate(num_points=500)
            data_with_id = {
                "x": data["x"].tolist(),
                "y": data["y"].tolist(),
            }
            save_new_json(data_with_id, f"{system_dir}/data.json")

            x = data["x"]
            y = data["y"]

            # Add noise to the generated data
            for noise_level in [0.05, 0.10, 0.15]:
                noise_adder = NoiseAdder(
                    noise_type="proportional", noise_level=noise_level
                )
                noisy_data = noise_adder.add_noise(y)
                noisy_data_with_id = {"data": noisy_data.tolist()}
                save_new_json(
                    noisy_data_with_id,
                    f"{system_dir}/data_with_noise_{int(noise_level * 100)}.json",
                )

                # Smooth the data with splines
                smoother = SplineSmoother(lambda_value=0.1)
                smoothed_data = smoother.smooth(x, noisy_data)
                smoothed_data_with_id = {"data": smoothed_data.tolist()}
                save_new_json(smoothed_data_with_id, f"{system_dir}/smoothed_data_{int(noise_level * 100)}.json")

                # Estimate parameters and save the results
                estimator = KKTLinearODEParameterEstimator(model, np.column_stack((x, smoothed_data.T)))
                estimated_params = estimator.solve()
                estimation_with_id = {"parameters": estimated_params}
                save_new_json(estimation_with_id, f"{system_dir}/parameter_estimations_{int(noise_level * 100)}.json")


def save_json(data, file_path):
    try:
        with open(file_path, "r") as f:
            existing_data = json.load(f)
    except Exception:
        existing_data = []

    existing_data.append(data)

def save_new_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    process_ode_systems(batch_size=100)
