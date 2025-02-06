import json
from statistics import median, mode

import numpy as np
from numpy import mean

from odeestimatorpy.data_generator.noise.noise_adder import NoiseAdder
from odeestimatorpy.data_generator.ode_integrator import ODEIntegrator
from odeestimatorpy.data_generator.spline_smoother import LambdaOptimizer
from odeestimatorpy.models.linear_ode_model import LinearODEModel

with open("D:\School\Tesis\ode-estimator\examples\odes_identifiable.json", "r") as file:
    systems = json.load(file)

models = [LinearODEModel.from_dict(system) for system in systems]

for model in models:
    model.set_generated_inputs()
    model.set_generated_parameters()
    model.set_generated_initial_conditions()

integrators = [(model, ODEIntegrator(model, [0, 20])) for model in models]

observations = [(model, integrator.integrate()) for model, integrator in integrators]

x = np.linspace(0, 20, 100)

noise_adder = NoiseAdder(noise_type="proportional")

observations_with_noise = [(model, observation, noise_adder.add_noise(observation)) for model, observation in observations]

optimized_lambdas = [(model, observation, noise_observation, LambdaOptimizer(x, noise_observation, observation).optimize_lambda()) for model, observation, noise_observation in observations_with_noise]

lambdas = [optimized_lambda for model, observation, noise_observation, optimized_lambda in optimized_lambdas]

with open("lambdas.json", 'w') as file:
    json.dump(lambdas, file)

print(f"Min: {min(lambdas)}")
print(f"Max: {max(lambdas)}")
print(f"Mean: {mean(lambdas)}")
print(f"Median: {median(lambdas)}")
print(f"Mode: {mode(lambdas)}")
