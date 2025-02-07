import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import make_splrep


class SplineSmoother:
    def __init__(self, s_value=0.1):
        """
        Initialize the spline smoother with a specific regularization parameter.

        Parameters:
            s_value (float): smoothing condition
        """
        self.s_value = s_value
        self.splines = None  # List of splines for each column of y

    def smooth(self, x, y):
        """
        Smooth the data using a spline with the current lambda value.

        Parameters:
            x (array-like): Independent variable values.
            y (array-like): Dependent variable values (data to be smoothed).

        Returns:
            smoothed_y (array): The smoothed values (same shape as y).
        """
        # Initialize a list to store the smoothed results for each column
        smoothed_y = np.zeros_like(y)

        for i in range(y.shape[0]):  # Iterate over columns (each dependent variable)
            spline = make_splrep(x, y[i, :], s=self.s_value)
            smoothed_y[i, :] = spline(x)  # Smooth each column separately

        return smoothed_y