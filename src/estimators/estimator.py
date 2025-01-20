from typing import Callable

import numpy as np

from abc import ABC, abstractmethod

from src.models.ode_model import ODEModel


class AbstractODESolver(ABC):
    def __init__(self, model: ODEModel, ode_results: np.ndarray, solver=Callable):
        """
        Initialize the solver for an ordinary differential equation system.

        Args:
            model (ODEModel): Ordinary differential equation system to estimate parameters
            ode_results (numpy.ndarray): Data matrix (rows: points, columns: variables)
            solver: (callable): Solver to use.
        """

        self.model = model
        self.ode_results = ode_results
        self.solver = solver

    @abstractmethod
    def solve(self):
        """Solve the ODE system and estimate parameters."""
        pass

    @abstractmethod
    def _build_system_matrix(self, normal_matrices, r_vectors, c_vectors):
        """Build the full system matrix."""
        pass

    @abstractmethod
    def _build_rhs_vector(self, size):
        """Construct the right-hand side vector for the system."""
        pass
