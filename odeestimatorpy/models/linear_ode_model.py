from typing import List, Dict

import sympy as sp

from odeestimatorpy.models.ode_model import ODEModel


class LinearODEModel(ODEModel):
    """
    Implementation of ODEModel for linear systems.
    """

    def __init__(self, equations: List[str], dependent_variables: List[str], independent_variable: str = "t",
                 initial_conditions: List[str] | Dict[str, float] = None,
                 parameters: List[str] | Dict[str, float] = None, parameters_names: List[str] = None,
                 constraints: List[str] = None, inputs: List[str] | Dict[str, float] = None,
                 outputs: List[str] = None, ):
        super().__init__(equations, dependent_variables, independent_variable, initial_conditions,
                         parameters, parameters_names, constraints, inputs, outputs)

        self.check_linearity()

    def _expand_and_analyze_terms(self, equation):
        """Expand and analyze terms in a given equation."""
        expanded_eq = sp.expand(equation)
        for term in expanded_eq.as_ordered_terms():
            yield term, [symbol for symbol in self.parameter_symbols if term.has(symbol)]

    @staticmethod
    def _verify_term_linearity(term, involved_params, equation_index):
        """Verify if a term is linear with respect to parameters."""
        for param in involved_params:
            if not sp.poly(term, param).is_linear:
                raise ValueError(f"Equation {equation_index + 1} is not linear in parameter '{param}': {term}")

    def check_linearity(self):
        """Checks if the system of equations is linear with respect to the parameters."""
        for i, equation in enumerate(self.equations):
            for term, involved_params in self._expand_and_analyze_terms(equation):
                if len(involved_params) > 1:
                    raise ValueError(f"Equation {i + 1} contains a nonlinear term involving multiple params: {term}")
                self._verify_term_linearity(term, involved_params, i)