from scipy.integrate import solve_ivp
import numpy as np

from odeestimatorpy.models.ode_model_base import ODEModelBase


class ODEIntegrator:
    """
    Handles the numerical integration of ODE systems and manages initial conditions.
    """

    def __init__(self, model: ODEModelBase, t_span = None, method="RK45"):
        """
        Initialize the integrator with the ODE system and parameters.

        Parameters:
            model (ODEModelBase): ODE model to use.
            t_span (tuple, optional): Value span for the integration (start, end).
            method (str): Integration method.
                options:
                   - "RK45" (default): Explicit Runge-Kutta method of order 5(4).
                   - "RK23": Explicit Runge-Kutta method of order 3(2).
                   - "DOP853": Explicit Runge-Kutta method of order 8.
                   - "Radau": Implicit Runge-Kutta method of the Radau IIA family of order 5
                   - "BDF": Implicit multistep variable-order (1 to 5) method based on a backward differentiation formula for the derivative approximation.
                   - "LSODA": Adams/BDF method with automatic stiffness detection and switching.
        """

        self.ode_system = model.compute_derivatives
        self.t_span = t_span
        self.initial_conditions = list(model.initial_conditions.values())
        self.method = method
        self.solution = None

    def integrate(self, t_eval=None, independent_variable_name="time", num_points=100):
        """
        Perform the integration of the ODE system with a customizable independent variable.

        Parameters:
            t_eval (array-like, optional): Specific points at which to evaluate the solution. If None, defaults to np.linspace over t_span.
            independent_variable_name (str): Name of the independent variable (e.g., "time", "space").
            num_points (int, optional): Number of points to evaluate the integration over.

        Returns:
            dict: A dictionary containing the independent variable points and corresponding ODE solutions.
        """
        if self.initial_conditions is None or len(self.initial_conditions) == 0:
            raise ValueError(
                "Initial conditions are not set. Use generate_initial_conditions() or provide them during initialization."
            )


        # Default to np.linspace if t_eval is not provided
        if t_eval is None:
            t_eval = np.linspace(self.t_span[0], self.t_span[1], num_points)
        else:
            t_eval = np.array(t_eval)  # Ensure t_eval is a NumPy array

            if self.t_span is not None:
                # Validate that t_eval lies within the integration range
                if t_eval[0] < self.t_span[0] or t_eval[-1] > self.t_span[1]:
                    raise ValueError(
                        f"Values in t_eval must be within the range defined by t_span {self.t_span}."
                    )

        # Perform the integration
        result = solve_ivp(
            self.ode_system,
            self.t_span,
            self.initial_conditions,
            method=self.method,
            t_eval=t_eval
        )

        if not result.success:
            raise RuntimeError("ODE integration failed!")

        # Store and return the solution
        self.solution = {
            "y": result.y,
            "x": t_eval
        }

        return self.solution

    def get_solution(self):
        """
        Retrieve the solution after integration.

        Returns:
            dict: The solution containing time points and values.
        """
        if self.solution is None:
            raise ValueError("No solution found. Please run integrate() first.")
        return self.solution
