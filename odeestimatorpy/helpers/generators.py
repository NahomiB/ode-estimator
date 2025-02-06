# General value generation methods
import numpy as np

def generate_random_values(num_variables, lower_bound=0, upper_bound=1):
    return np.random.uniform(lower_bound, upper_bound, num_variables).tolist()

def generate_zero_values(num_variables):
    return [0.0] * num_variables

def generate_constant_values(num_variables, constant_value=1.0):
    return [constant_value] * num_variables

def generate_normal_values(num_variables, mean=0, std_dev=1):
    return np.random.normal(mean, std_dev, num_variables).tolist()

def generate_linear_values(num_variables, start=0, step=1):
    return [start + i * step for i in range(num_variables)]

def generate_boundary_values(num_variables, lower_bound=0, upper_bound=1):
    return [lower_bound if i % 2 == 0 else upper_bound for i in range(num_variables)]

def generate_logarithmic_values(num_variables, lower_bound=0.1, upper_bound=10):
    return np.logspace(np.log10(lower_bound), np.log10(upper_bound), num=num_variables).tolist()

def generate_perturbed_values(base_values, perturbation_scale=0.1):
    return [
        val + np.random.uniform(-perturbation_scale, perturbation_scale)
        for val in base_values
    ]