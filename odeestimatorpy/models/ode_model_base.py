from abc import ABC, abstractmethod
from sympy import Eq, Lt, Gt, Le, Ge, Expr, Symbol, symbols
from typing import List, Dict, Callable, Union

# Constraint types could be Eq, Lt, Gt, Le, Ge from sympy.
Constraint = Union[Eq, Lt, Gt, Le, Ge]


class ODEModelBase(ABC):
    """
    Abstract base class to define the structure and types for an ODE model.
    """

    def __init__(self):
        # Define properties with specific types
        self.equations: List[Expr] = []  # List of equation expressions
        self.functions: List[Callable] = []  # List of equation functions

        self.variables: List[str] = []   # List of variable names
        self.variable_symbols: List[Symbol] = []  # List of variable symbols
        self.independent_variable: Symbol = symbols("t")

        self.parameter_names: List[str] = [] #List of parameters names
        self.parameter_symbols: List[Symbol] = []  # List of parameter symbols
        self.parameters: Dict[str, float] = {}  # Dictionary of parameter names and values

        self.inputs: Dict[str, float] = {} # Dictionary of inputs names and values
        self.inputs_symbols: List[Symbol] = []  # List of inputs symbols

        self.outputs: List[Eq] = []

        self.constraints: List[Constraint] = []  # List of constraint objects (Eq, Lt, Gt, Le, Ge)
        self.initial_conditions: Dict[str, float] = {} # Dictionary of initial conditions names and values

    @abstractmethod
    def set_equations(self, equations) -> None:
        """
        Abstract method to set the equations for the model.
        """
        pass

    @abstractmethod
    def set_variables(self, variables, independent_variable) -> None:
        """
        Abstract method to set the variables for the model.
        """
        pass

    @abstractmethod
    def set_parameters(self, parameters, parameters_names) -> None:
        """
        Abstract method to set the parameters for the model.
        """
        pass

    @abstractmethod
    def set_constraints(self, constraints) -> None:
        """
        Abstract method to set the constraints for the model.
        """
        pass

    @abstractmethod
    def set_initial_conditions(self, constraints) -> None:
        """
        Abstract method to set the constraints for the model.
        """
        pass

    @abstractmethod
    def set_inputs(self, inputs):
        """
        Abstract method to set the inputs for the model.
        """
        pass

    @abstractmethod
    def set_outputs(self, outputs):
        """
        Abstract method to set the outputs for the model.
        """
        pass

    @abstractmethod
    def evaluate(self, y: List[float]) -> List[float]:
        """
        Abstract method to evaluate the system of ODEs at a given point.
        """
        pass

    def compute_derivatives(self, dependent_values: List[float], independent_value: float) -> List[float]:
        """
        Abstract method to compute the derivatives of the system of ODEs at a set of given points.
        """
        pass


