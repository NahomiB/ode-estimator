import numpy as np
import pyswarms as ps

from odeestimatorpy.data_generator.ode_integrator import ODEIntegrator
from odeestimatorpy.estimators.estimator import AbstractODEEstimator


class PSOEstimator(AbstractODEEstimator):
    """
    Class for estimating parameters of an ordinary differential equation (ODE) system
    using Particle Swarm Optimization (PSO).
    """

    def __init__(self, model, ode_results, n_particles=40, iters=1000, options=None, tolerance=1e-5, patience=10):
        """
        Initialize the estimator with the necessary parameters.

        Args:
            model (ODEModel): Ordinary differential equation system to estimate parameters
            ode_results (numpy.ndarray): Data matrix (rows: points, columns: variables)
            n_particles (int): Number of particles in PSO.
            iters (int): Number of iterations in PSO.
            options (dict, optional): PSO algorithm parameters ('c1', 'c2', 'w').
            tolerance(float, optional): Defines the minimum improvement in the cost function considered significant.
            patience (int, optional): Specifies the number of consecutive iterations with an improvement below tolerance before stopping.
        """

        super().__init__(model, ode_results)

        self.t_eval = ode_results[:, 0].T
        self.num_points = ode_results.shape[0]
        self.y = ode_results[:, 1:]

        self.n_particles = n_particles
        self.iters = iters
        self.options = options if options else {'c1': 0.3, 'c2': 0.5, 'w': 0.9}
        self.tolerance = tolerance
        self.patience = patience

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

            mse = np.mean((y_pred.T - self.y) ** 2)  # Mean squared error
            costs[i] = mse

        return costs


    def solve(self):
        """
        Run the PSO algorithm to find the best parameters that minimize the cost function.

        Returns:
            numpy.ndarray: Best set of estimated parameters found by PSO.
        """
        optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=len(self.model.parameter_names), options=self.options)

        best_cost_history = []
        no_improve_count = 0  # Counter for stagnation

        for i in range(self.iters):
            best_cost, best_params = optimizer.optimize(self.cost_function, iters=1, verbose=True)
            best_cost_history.append(best_cost)

            # Check for early stopping
            if i > 0 and abs(best_cost_history[-1] - best_cost_history[-2]) < self.tolerance:
                no_improve_count += 1
            else:
                no_improve_count = 0  # Reset counter if improvement occurs

            if no_improve_count >= self.patience:
                break  # Stop optimization early

        self.best_cost, self.best_params = best_cost, best_params

        parameters_by_name = {name: best_params[index] for index, name in enumerate(self.model.parameter_names)}
        return parameters_by_name
