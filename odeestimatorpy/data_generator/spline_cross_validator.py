import numpy as np
from scipy.interpolate import make_splrep, splev
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from concurrent.futures import ThreadPoolExecutor


class SplineCrossValidator:
    """
    A class to perform k-fold cross-validation on a smoothing spline model.
    """

    def __init__(self, independent_values, dependent_values, k=5):
        """
        Initialize the cross-validator with input data.

        Args:
            independent_values (np.ndarray): Array of input values.
            dependent_values (np.ndarray): Array of output values.
            k (int): Number of folds for cross-validation.
        """
        self.independent_values = np.asarray(independent_values)
        self.dependent_values = np.asarray(dependent_values)
        self.k = k

    def _validate_fold(self, train_index, val_index, s):
        """
        Evaluate a specific fold by training and validating the model with a given smoothing parameter.

        Args:
            train_index (np.ndarray): Indices for training.
            val_index (np.ndarray): Indices for validation.
            s (float): Smoothing parameter for spline interpolation.

        Returns:
            float: Root mean squared error (RMSE) for the fold.
        """

        x_train, x_val = self.independent_values[train_index], self.independent_values[val_index]
        y_train, y_val = self.dependent_values[:,train_index], self.dependent_values[:,val_index]

        y_pred = np.zeros_like(y_val)

        for i in range(y_train.shape[0]):  # Iterate over columns (each dependent variable)
            spline = make_splrep(x_train, y_train[i, :], s=s)
            y_pred[i, :] = splev(x_val, spline)

        return np.sqrt(mean_squared_error(y_val, y_pred))

    def _cross_validate_for_s(self, s):
        """
        Perform k-fold cross-validation for a given smoothing parameter.

        Args:
            s (float): Smoothing parameter.

        Returns:
            Tuple[float, List[float]]: Mean RMSE and list of RMSE values for each fold.
        """
        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)

        with ThreadPoolExecutor() as executor:
            result = [ executor.submit(self._validate_fold, train_index, val_index, s) for train_index, val_index in kf.split(self.independent_values)]

        errors = [future.result() for future in result]
        return np.mean(errors), errors

    def cross_validate(self, s_eval):
        """
        Perform cross-validation for multiple smoothing values.

        Args:
            s_eval (List[float]): List of smoothing values to evaluate.

        Returns:
            Dict[float, Tuple[float, List[float]]]: Dictionary mapping each smoothing value to its mean error and individual fold errors.
        """
        with ThreadPoolExecutor() as executor:
           result = {s: executor.submit(self._cross_validate_for_s, s) for s in s_eval}

        return {s: future.result() for s, future in result.items()}

    @staticmethod
    def get_best_s(means):
        """
        Find the best smoothing value based on the lowest mean error.

        Args:
            means (Dict[float, float]): Dictionary mapping each smoothing value to its mean error.

        Returns:
            Tuple[float, float]: The best smoothing value and its corresponding error.
        """
        best = min(means, key=means.get)
        return best, means[best]