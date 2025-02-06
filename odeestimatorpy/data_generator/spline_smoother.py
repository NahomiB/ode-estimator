import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import make_splrep


class SplineSmoother:
    def __init__(self, lambda_value=0.1):
        """
        Initialize the spline smoother with a specific regularization parameter.

        Parameters:
            lambda_value (float): Regularization parameter (lambda). Controls the smoothness of the spline.
        """
        self.lambda_value = lambda_value
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
            spline = make_splrep(x, y[i, :], s=self.lambda_value)
            smoothed_y[i, :] = spline(x)  # Smooth each column separately

        return smoothed_y


class LambdaOptimizer:
    def __init__(self, x, y, original_y, lambda_range=(0.01, 0.1), max_iterations=100):
        """
        Initialize the lambda optimizer.

        Parameters:
            x (array-like): Independent variable values (data points).
            y (array-like): Dependent variable values (data points).
            lambda_range (tuple): The range (min, max) to search for optimal lambda.
            max_iterations (int): Maximum number of iterations for optimization.
        """
        self.x = x
        self.y = y
        self.original_y = original_y
        self.lambda_range = lambda_range
        self.max_iterations = max_iterations

    def objective_function(self, lambda_value):
        """
        Objective function for optimization. It computes the error (e.g., MSE) for a given lambda.

        Parameters:
            lambda_value (float): The value of lambda to be tested.

        Returns:
            float: The computed error for this lambda.
        """
        # Create a spline smoother and smooth all columns of y

        smoothed_y = np.zeros_like(self.y)

        for i in range(self.y.shape[0]):  # Iterate over columns (each dependent variable)
            spline = make_splrep(self.x, self.y[i, :], s=lambda_value)
            smoothed_y[i, :] = spline(self.x)  # Smooth each column separately

        # Compute the Mean Squared Error (MSE) for all columns
        error = np.mean((self.original_y - smoothed_y) ** 2)  # MSE for all columns
        return error

    def optimize_lambda(self):
        """
        Optimize lambda by minimizing the error between the data and smoothed values.

        Returns:
            float: The optimal lambda value.
        """
        result = minimize(
            self.objective_function,
            x0=np.array(np.mean(self.lambda_range)),  # Initial guess (mean of the range)
            bounds=[self.lambda_range],
            options={"maxiter": self.max_iterations}
        )
        return result.x[0]
