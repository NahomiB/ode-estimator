import numpy as np

from odeestimatorpy.data_generator.noise.noise_generator import NoiseGenerator


class UniformNoiseGenerator(NoiseGenerator):
    """
    Generates uniform noise.
    """
    def generate_noise(self, size):
        return np.random.uniform(0, self.noise_level, size)
