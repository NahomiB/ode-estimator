import json
import matplotlib.pyplot as plt
import os
import yaml
from scipy.integrate import odeint

from sympy import symbols, Eq, sympify, lambdify
from typing import List, Dict, Tuple

from src.models.ode_model_base import Constraint, ODEModelBase


class ODEModel(ODEModelBase):
    """
    Concrete implementation of ODEModel based on the abstract base class ODEModelBase.
    """

    def __init__(self, equations: List[str], variables: List[str], parameters: Dict[str, float],
                 constraints: List[str] = None, initial_conditions: List[str] | Dict[str, float] = None):
        super().__init__()

        # Initialize the properties using the setter methods
        self.set_variables(variables)
        self.set_parameters(parameters)
        self.set_constraints(constraints or [])
        self.set_equations(equations)

        if initial_conditions:
            self.set_initial_conditions(initial_conditions)

    def set_equations(self, equations: List[str]) -> None:
        """
        Set the equations for the model and convert them to callable functions.

        Args:
            equations (List[str]): List of string representations of the equations.
        """
        all_symbols = symbols(self.variables + list(self.parameters.keys()))

        self.equations = [sympify(eq, evaluate=False) for eq in equations]
        self.functions = [lambdify(all_symbols, expr) for expr in self.equations]

    def set_variables(self, variables: List[str]) -> None:
        """
        Set the variables for the model.

        Args:
            variables (List[str]): List of variables in the ODE model.
        """
        self.variables = variables

    def set_parameters(self, parameters: Dict[str, float]) -> None:
        """
        Set the parameters for the model.

        Args:
            parameters (Dict[str, float]): Dictionary of parameters and their values.
        """
        self.parameters = parameters

    def set_constraints(self, constraints: List[str]) -> None:
        """
        Set the constraints for the model by converting a list of string constraints into sympy relational expressions.

        Args:
            constraints (List[str]): List of constraints as strings.
        """
        # Convert each constraint string to the corresponding sympy relational expression
        self.constraints = [self.parse_constraint(c) for c in constraints]

    def set_initial_conditions(self, initial_conditions) -> None:
        """
        Set the initial conditions for the dependent variables, either from a dictionary or a list of strings.

        Args:
            initial_conditions (Dict[str, float] or List[str]): Initial conditions as strings or as a dictionary of
                variable-value pairs.
        """
        if isinstance(initial_conditions, list):
            # Parse the list of string initial conditions
            self.initial_conditions = {self._parse_initial_condition(cond) for cond in initial_conditions}
        elif isinstance(initial_conditions, dict):
            # Use the provided dictionary of initial conditions directly
            self.initial_conditions = initial_conditions
        else:
            raise ValueError("Initial conditions must be either a dictionary or a list of strings.")

    @classmethod
    def _load_model_data(cls, data):
        """
        Helper function to load and parse the common model data (equations, variables, parameters, constraints).

        Args:
            data (dict): The data dictionary with keys 'equations', 'variables', 'parameters', and 'constraints'.

        Returns:
            tuple: A tuple containing (equations, variables, parameters, constraints).

        Raises:
            ValueError: If any required data is missing or invalid.
        """

        # Check if the required keys are present in the data
        required_keys = ['equations', 'variables']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        equations = [sympify(eq, evaluate=False) for eq in data['equations']]
        variables = [symbols(var) for var in data['variables']]
        parameters = data.get('parameters', {})
        constraints = data.get('constraints', [])
        return equations, variables, parameters, constraints

    @classmethod
    def from_json(cls, file_path):
        """
        Alternative constructor to initialize the ODE model from a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            ODEModel: Initialized model object.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is invalid.
        """
        # Check if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file at path {file_path} does not exist.")

        # Load data from the JSON file
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                raise ValueError(f"The file at {file_path} is not a valid JSON file.")

        equations, variables, parameters, constraints = cls._load_model_data(data)
        return cls(equations, variables, parameters, constraints)

    @classmethod
    def from_yaml(cls, file_path):
        """
        Alternative constructor to initialize the ODE model from a YAML file.

        Args:
            file_path (str): Path to the YAML file.

        Returns:
            ODEModel: Initialized model object.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is invalid.
        """
        # Check if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file at path {file_path} does not exist.")

        # Load data from the YAML file
        with open(file_path, 'r') as file:
            try:
                data = yaml.safe_load(file)
            except yaml.YAMLError:
                raise ValueError(f"The file at {file_path} is not a valid YAML file.")

        equations, variables, parameters, constraints = cls._load_model_data(data)
        return cls(equations, variables, parameters, constraints)

    @classmethod
    def from_dict(cls, data):
        """
        Alternative constructor to initialize the ODE model directly from a dictionary.

        Args:
            data (dict): A dictionary containing the ODE model data, including 'equations',
                         'variables', 'parameters', and 'constraints'.

        Returns:
            ODEModel: Initialized model object.
        """
        equations, variables, parameters, constraints = cls._load_model_data(data)
        return cls(equations, variables, parameters, constraints)

    @staticmethod
    def parse_constraint(constraint_str: str) -> Constraint:
        """
        Parse an equality or inequality string and return the corresponding sympy relational expression.

        Args:
            constraint_str (str): The constraint as a string (e.g., 'x + y = 10', 'x - y < 5', 'z >= 3').

        Returns:
            sympy relational expression: Corresponding sympy expression (Eq, Lt, Gt, Le, Ge).

        Raises:
            ValueError: If the string format is invalid.
        """
        constraint = sympify(constraint_str, evaluate=False)

        if isinstance(constraint, Constraint):
            return constraint
        else:
            raise ValueError(f"Invalid constraint: {constraint}. Must be an equality or inequality.")

    @staticmethod
    def _parse_initial_condition(condition: str) -> Tuple[str, float]:
        """
        Parse an initial condition string into a sympy expression and extract the variable and value.

        Args:
            condition (str): Initial condition as a string, e.g., 'x = 1' or 'y = 0'.

        Returns:
            tuple: A tuple (variable, value) representing the initial condition.
        """
        condition = condition.replace(' ', '')
        eq = sympify(condition)

        if isinstance(eq, Eq):
            return eq.lhs, eq.rhs
        else:
            raise ValueError(f"Invalid initial condition: {condition}. Must be of the form 'var = value'.")

    @staticmethod
    def graph(independent_var, dependent_vars, title: str = "ODE System Solution",
              x_label: str = "Independent Variable", y_label: str = "Dependent Variables",
              sample_rate: int = 1, separate_plots: bool = False) -> None:
        """
        Plot the solution of the ODE system.

        Args:
            independent_var (array-like): Array-like object representing the independent variable (e.g., time, space,
                or any other variable). It should be a 1D numpy array or list containing the values for the independent
                variable.

            dependent_vars (list of array-like): A list of 1D numpy arrays or lists, where each array represents a
                solution (dependent variable) of the ODE system. Each array must have the same length as the independent
                variable.

            title (str, optional): The title of the plot. Default is 'ODE System Solution'. This title will appear at
                the top of the graph.

            x_label (str, optional): Label for the x-axis. Default is 'Independent Variable'. This label should describe
                the independent variable.

            y_label (str, optional): Label for the y-axis. Default is 'Dependent Variables'. This label will apply to
                all dependent variable subplots.

            sample_rate (int, optional): Indicates the rate at which data points should be sampled. For example,
                a value of `2` will display every second point in the dataset. Default is 1 (no sampling). Sampling
                helps optimize the display for large datasets.

            separate_plots (bool, optional): Whether to plot each dependent variable in its own subplot. If False,
                all will be plotted on the same axis. Defaults to False.

        Returns:
            None: This method displays the plot directly using Matplotlib. It does not return any value.

        Raises:
            ValueError: If the lengths of the dependent variables do not match the length of the independent variable.

        """

        # Validate dimensions
        if not all(len(independent_var) == len(dep_var) for dep_var in dependent_vars):
            raise ValueError("All dependent variables must have the same length as the independent variable.")

        # Sample the data if the dataset is large
        if sample_rate > 1:
            sampled_data = slice(None, None, sample_rate)
            independent_var = independent_var[sampled_data]
            dependent_vars = [dep_var[sampled_data] for dep_var in dependent_vars]

        if separate_plots:
            # If we are using separate subplots, create one plot per dependent variable
            num_vars = len(dependent_vars)
            fig, axes = plt.subplots(num_vars, 1, figsize=(10, 6 * num_vars), facecolor='#f9f9f9')

            # If only one variable, make axes an iterable
            if num_vars == 1:
                axes = [axes]

            # Plot each dependent variable in its own subplot
            for ax, dep_var in zip(axes, dependent_vars):
                ax.plot(independent_var, dep_var, alpha=0.7, lw=2)
                ax.set_xlabel(x_label, fontsize=12)
                ax.set_ylabel(y_label, fontsize=12)
                ax.set_title(f'{title} - {y_label}', fontsize=14, weight='bold')
                ax.grid(True, which='major', linestyle='-', linewidth=0.8, color='gray', alpha=0.6)
                ax.yaxis.set_tick_params(length=0)
                ax.xaxis.set_tick_params(length=0)

                # Remove the spines for a cleaner look
                for spine in ('top', 'right', 'bottom', 'left'):
                    ax.spines[spine].set_visible(False)

        else:
            # If not using separate subplots, plot all dependent variables on the same graph
            plt.figure(figsize=(10, 6))
            for dep_var in zip(dependent_vars):
                plt.plot(independent_var, dep_var, alpha=0.7, lw=2)

            plt.xlabel(x_label, fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.title(f'{title} - {y_label}', fontsize=14, weight='bold')
            plt.grid(True, which='major', linestyle='-', linewidth=0.8, color='gray', alpha=0.6)
            plt.legend([f'Variable {i + 1}' for i in range(len(dependent_vars))], loc='best')

        # Adjust the layout for better spacing
        plt.tight_layout()

        # Show the plot
        plt.show()

    def evaluate(self, y: List[float]) -> List[float]:
        """
        Evaluate the ODE system at a given point using sympy expressions.

        Args:
            y (List[float]): List of values for the dependent variables (and possibly time).

        Returns:
            List[float]: List of derivatives at the given point.
        """
        if len(y) != len(self.variables):
            raise ValueError("Mismatch between number of variables and input values.")

        # Map variables and parameters to their corresponding values
        values: dict[str, float] = {var: val for var, val in zip(self.variables, y)}
        values.update({param: val for param, val in self.parameters.items()})

        # Evaluate each equation by substituting the values into the sympy expressions
        results = [eq.subs(values).evalf() for eq in self.equations]

        return results

    def compute_derivatives(self, dependent_values: List[float], independent_value: float):
        """
            Calculates the derivatives of the ODE system at a given point using the defined functions.

            Args:
                dependent_values: Current values of the dependent variables in the system.
                independent_value: The current value of the independent variable (e.g., time).

            Returns:
                A list of the derivatives of the system evaluated at the given point.
        """

        return [f(independent_value, *dependent_values) for f in self.functions]

    def integrate_system(self, independent_values: List[float]):
        """
        Integrates the system of ODEs using a numerical integration method (odeint).

        Args:
            independent_values: A list of values of the independent variable over time.

        Returns:
            The result of integrating the system of differential equations.
        """

        return odeint(self.compute_derivatives, self.initial_conditions, independent_values)
