class NoiseGenerator:
    """
    Base class for generating and applying noise to data.
    """
    def __init__(self, noise_level=0.1):
        """
        Initialize the noise generator with a specific level and custom noise function.

        Parameters:
            noise_level (float, optional): Magnitude of the noise. Defaults to 0.1.
        """
        self.noise_level = noise_level

    def generate_noise(self, size):
        """
        Method to generate noise. This should be overridden in subclasses.

        Parameters:
            size: The size of the generated noise.

        Returns:
            np.ndarray: The generated noise.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

