import numpy as np

from odeestimatorpy.data_generator.noise.noise_generator import NoiseGenerator


class NormalNoiseGenerator(NoiseGenerator):
    """
    Generates normal (Gaussian) noise.
    """
    def generate_noise(self, size):
        return np.random.normal(0, self.noise_level, size)