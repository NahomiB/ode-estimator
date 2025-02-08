from typing import Callable

import numpy as np

from abc import ABC, abstractmethod

from odeestimatorpy.models.ode_model_base import ODEModelBase


class AbstractODEEstimator(ABC):
    def __init__(self, model: ODEModelBase, ode_results: np.ndarray):
        """
        Initialize the solver for an ordinary differential equation system.

        Args:
            model (ODEModel): Ordinary differential equation system to estimate parameters
            ode_results (numpy.ndarray): Data matrix (rows: points, columns: variables)
        """

        self.model = model
        self.ode_results = ode_results

    @abstractmethod
    def solve(self):
        """Solve the ODE system and estimate parameters."""
        pass
