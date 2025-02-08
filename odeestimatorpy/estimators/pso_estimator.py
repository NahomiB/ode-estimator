import numpy as np
import pyswarms as ps

from odeestimatorpy.data_generator.ode_integrator import ODEIntegrator
from odeestimatorpy.estimators.estimator import AbstractODEEstimator

class PSOEstimator(AbstractODEEstimator):
    """
    Class for estimating parameters of an ordinary differential equation (ODE) system
    using Particle Swarm Optimization (PSO).
    """

    def __init__(self, model, ode_results, n_particles=30, iters=100, options=None):
        """
        Initialize the estimator with the necessary parameters.

        Args:
            model (ODEModel): Ordinary differential equation system to estimate parameters
            ode_results (numpy.ndarray): Data matrix (rows: points, columns: variables)
            n_particles (int): Number of particles in PSO.
            iters (int): Number of iterations in PSO.
            options (dict, optional): PSO algorithm parameters ('c1', 'c2', 'w').
        """

        super().__init__(model, ode_results)

        self.t_eval = ode_results[:, 0].T
        self.num_points = ode_results.shape[0]
        self.y = ode_results[:, 1:]

        self.n_particles = n_particles
        self.iters = iters
        self.options = options if options else {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
        self.best_params = None
        self.best_cost = None


    def cost_function(self, params):
        """
        Cost function based on the mean squared error between real and simulated data.

        Args:
            params (numpy.ndarray): Set of parameters to evaluate. Each row represents a particle.

        Returns:
            numpy.ndarray: Cost values for each particle.
        """
        num_particles = params.shape[0]
        costs = np.zeros(num_particles)

        for i in range(num_particles):

            parameters = params[i]
            parameters_by_name = {name: parameters[index] for index, name in enumerate(self.model.parameter_names)}
            self.model.set_parameters(parameters_by_name, [])

            integrator = ODEIntegrator(self.model)
            y_pred =  integrator.integrate(self.t_eval, self.num_points)["y"]

            mse = np.mean((y_pred - self.y) ** 2)  # Mean squared error
            costs[i] = mse

        return costs

    def solve(self):
        """
        Run the PSO algorithm to find the best parameters that minimize the cost function.

        Returns:
            numpy.ndarray: Best set of estimated parameters found by PSO.
        """
        optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=len(self.model.parameter_names), options=self.options)
        self.best_cost, self.best_params = optimizer.optimize(self.cost_function, iters=self.iters)
        return self.best_params
