import numpy as np

from odeestimatorpy.data_generator.noise.noise_generator import NoiseGenerator
from odeestimatorpy.data_generator.noise.normal_noise_generator import NormalNoiseGenerator


class NoiseAdder:
    """
    Adds noise to data, supporting additive and proportional noise models.
    """

    def __init__(self, noise_type="additive", noise_level=0.1, noise_generator=NormalNoiseGenerator):
        """
        Initialize the noise adder with a specific type and level.

        Parameters:
            noise_type (str, optional): Type of noise to apply ("additive" or "proportional"). Defaults to "additive".
            noise_level (float, optional): Magnitude of the noise. Defaults to 0.1.
            noise_generator (NoiseGenerator, optional): A subclass of NoiseGenerator. Defaults to NormalNoiseGenerator.
        """

        if noise_type not in ["additive", "proportional"]:
            raise ValueError("Invalid noise_type. Choose 'additive' or 'proportional'.")
        if noise_level < 0:
            raise ValueError("Noise level must be non-negative.")

        self.noise_generator : NoiseGenerator = noise_generator(noise_level=noise_level)

        self.noise_type = noise_type
        self.noise_level = noise_level

    def add_noise(self, data):
        """
        Apply noise to the given data.

        Parameters:
            data (array-like): The data to which noise will be added.

        Returns:
            np.ndarray: The noisy data.
        """
        data = np.array(data)
        noise = np.array(self.noise_generator.generate_noise(data.shape))

        if self.noise_type == "additive":
            return data + noise

        elif self.noise_type == "proportional":
            return data + noise * data

        else:
            raise ValueError("Invalid noise_type. Choose 'additive' or 'proportional'.")

    def set_noise_type(self, noise_type):
        """
        Update the noise type.

        Parameters:
            noise_type (str): New noise type ("additive" or "proportional").
        """
        if noise_type not in ["additive", "proportional"]:
            raise ValueError("Invalid noise_type. Choose 'additive' or 'proportional'.")
        self.noise_type = noise_type

    def set_noise_level(self, noise_level):
        """
        Update the noise level.

        Parameters:
            noise_level (float): New noise level (must be non-negative).
        """
        if noise_level < 0:
            raise ValueError("Noise level must be non-negative.")
        self.noise_level = noise_level
