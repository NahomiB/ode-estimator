import numpy as np

from odeestimatorpy.data_generator.noise.noise_generator import NoiseGenerator


class LogarithmicNoiseGenerator(NoiseGenerator):
    """
    Generates logarithmic noise.
    """
    def generate_noise(self, size):
        lower_bound = self.noise_level
        upper_bound = self.noise_level * 10
        return np.logspace(np.log10(lower_bound), np.log10(upper_bound), size)
