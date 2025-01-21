from abc import ABC, abstractmethod
from sympy import Eq, Lt, Gt, Le, Ge, Expr, Symbol
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

        self.parameter_names: List[str] = [] #List of parameters names
        self.parameter_symbols: List[Symbol] = []  # List of parameter symbols
        self.parameters: Dict[str, float] = {}  # Dictionary of parameter names and

        self.constraints: List[Constraint] = []  # List of constraint objects (Eq, Lt, Gt, Le, Ge)
        self.initial_conditions: Dict[str, float] = {} # Dictionary of initial conditions names and values

    @abstractmethod
    def set_equations(self, equations) -> None:
        """
        Abstract method to set the equations for the model.
        """
        pass

    @abstractmethod
    def set_variables(self, variables) -> None:
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
    def evaluate(self, y: List[float]) -> List[float]:
        """
        Abstract method to evaluate the system of ODEs at a given point.
        """
        pass

    def integrate_system(self, independent_values: List[float]):
        """
        Integrates the system of ODEs.
        """
        pass
