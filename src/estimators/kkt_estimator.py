import numpy as np
import sympy as sp

from scipy.linalg import block_diag
from sympy import Eq, symbols, lambdify

from src.estimators.estimator import AbstractODESolver
from src.models.ode_model import ODEModel


class KKTLinearODEParameterEstimator(AbstractODESolver):
    def __init__(self, model: ODEModel, ode_results: np.ndarray, solver=np.linalg.solve):
        """
        Initialize the solver for a linear ordinary differential equation system.

        Args:
            model (ODEModel): Ordinary differential equation system to estimate parameters
            ode_results (numpy.ndarray): Data matrix without noise (rows: points, columns: variables)
            solver: (callable, optional): Solver to use. Defaults to np.linalg.solve
        """

        super().__init__(model, ode_results, solver)

        self.number_of_constraints = len(model.constraints)
        self.number_of_parameters = len(model.parameters)

        self.all_variables = symbols(model.variables)
        self.all_parameters = symbols(list(model.parameters.keys()))

        self.parameters_name = list(model.parameters.keys())

        self.callables_per_equation = [self._extract_callables(equation) for equation in self.model.equations]

        self.check_linearity()

    def _expand_and_analyze_terms(self, equation):
        """Expand and analyze terms in a given equation."""
        expanded_eq = sp.expand(equation)
        for term in expanded_eq.as_ordered_terms():
            yield term, [symbol for symbol in self.all_parameters if term.has(symbol)]

    @staticmethod
    def _verify_term_linearity(term, involved_params, equation_index):
        """Verify if a term is linear with respect to parameters."""
        for param in involved_params:
            if not sp.poly(term, param).is_linear:
                raise ValueError(f"Equation {equation_index + 1} is not linear in parameter '{param}': {term}")

    def check_linearity(self):
        """Checks if the system of equations is linear with respect to the parameters."""
        for i, equation in enumerate(self.model.equations):
            for term, involved_params in self._expand_and_analyze_terms(equation):
                if len(involved_params) > 1:
                    raise ValueError(f"Equation {i + 1} contains a nonlinear term involving multiple params: {term}")
                self._verify_term_linearity(term, involved_params, i)

    def _extract_callables(self, expr):
        """
        Given a sympy expression, returns a list of functions
        independent for each term multiplied by a parameter.

        Args:
            expr (sympy.Expr): The symbolic expression.

        Returns:
            list: List of independent callable functions.
        """
        # Ensure the expression is fully expanded
        expr = expr.expand()

        # Extract the terms of the expression
        terms = expr.as_ordered_terms()

        # List to store the callables
        callables = []

        # Iterate over the terms
        for term in terms:
            # Check if the term contains a parameter
            for param in self.all_parameters:
                if param in term.free_symbols:
                    factor = term / param
                    callables.append(lambdify(self.all_variables, factor, modules=["numpy"]))
                    break

        return callables

    @staticmethod
    def _compute_weighted_derivative(col, f, data):
        """
        Computes a weighted sum of function values evaluated over a dataset D
        where the weights are derived from the finite differences of the dataset.

        Args:
            col (int): The column index used to compute the finite differences (slope).
            f (callable): The function to evaluate.
            data (ndarray): Dataset of tuples, each containing values corresponding to the columns.

        Returns:
            float: The weighted sum of function values.
        """
        slopes = np.diff(data[:, col]) / np.diff(data[:, 0])
        f_values = np.array([f(*row) for row in data[:-1]])
        return np.sum(f_values * slopes)

    @staticmethod
    def _scalar_product(f, g, data):
        """
        Computes the scalar product of two functions over a dataset D using numpy.dot.

        Args:
            f (callable): The first function.
            g (callable): The second function.
            data (ndarray): An array of tuples representing the dataset. Each tuple contains input values.

        Returns:
            float: The scalar product result.
        """
        # Evaluate f and g over the dataset D
        f_values = np.array([f(*data_point) for data_point in data])
        g_values = np.array([g(*data_point) for data_point in data])

        # Use numpy.dot to compute the scalar product
        return np.dot(f_values, g_values)

    @staticmethod
    def _compute_normal_matrix(basis_functions, data):
        """
        Construct the normal matrix associated with a specific equation.

        Args:
            basis_functions (list[Callable]): List of basis functions for the equation.
            data (ndarray): An array of tuples representing the dataset. Each tuple contains input values.

        Returns:
            numpy.ndarray: Normal matrix N_fi.
        """

        number_of_parameters = len(basis_functions)
        normal_matrix = np.zeros((number_of_parameters, number_of_parameters))
        for j in range(number_of_parameters):
            for k in range(number_of_parameters):
                scalar_product = KKTLinearODEParameterEstimator._scalar_product(basis_functions[j], basis_functions[k],
                                                                                data)
                normal_matrix[j, k] = normal_matrix[k, j] = scalar_product

        return normal_matrix

    def _build_constraints(self):
        """
        Build the rij and cij vectors for equality constraints.

        Ensures all constraints are of the form x = y.
        Raises an exception if not.

        Returns:
            tuple: (dict, dict)
                - r_vectors: Dictionary of rij row vectors.
                - c_vectors: Dictionary of cij column vectors.
        """
        r_vectors = {}
        c_vectors = {}

        for constraint in self.model.constraints:
            # Ensure the constraint is an equality constraint
            if not isinstance(constraint, Eq):
                raise ValueError(f"Constraint {constraint} is not an equality constraint of the form x = y.")

            # Extract left-hand side and right-hand side
            lhs = constraint.lhs
            rhs = constraint.rhs

            # Ensure both lhs and rhs are valid parameters
            if lhs.name not in self.parameters_name or rhs.name not in self.parameters_name:
                raise ValueError(f"Both sides of the constraint {constraint} must be valid parameter names.")

            # Get the indices of the parameters
            i = self.parameters_name.index(lhs.name)
            j = self.parameters_name.index(rhs.name)

            # Build rij and cij vectors
            ri = np.zeros(self.number_of_parameters)
            ri[i] = 1
            ri[j] = -1

            r_vectors[(i, j)] = ri
            c_vectors[(i, j)] = ri.T

        return r_vectors, c_vectors

    def _build_system_matrix(self, normal_matrices, r_vectors, c_vectors):
        """
        Construct the system matrix A by combining normal matrices and constraints.

        Args:
            normal_matrices (list[numpy.ndarray]): List of normal matrices for each equation.
            r_vectors (dict[Tuple[int, int], numpy.ndarray]): List of rij row vectors.
            c_vectors (dict[Tuple[int, int], numpy.ndarray]): List of cij column vectors.

        Returns:
            numpy.ndarray: System matrix A.
        """
        blocks = normal_matrices + [
            np.zeros((self.number_of_constraints, self.number_of_constraints))
        ]

        a = block_diag(*blocks)

        # Start placing rij rows and cij columns in the corresponding positions
        index = a.shape[0] - self.number_of_constraints

        for (i, j), rij in r_vectors.items():
            a[index, :rij.shape[0]] = rij.flatten()

            cij = c_vectors[(i, j)]
            a[:cij.shape[0], index] = cij.flatten()  # Place the cij vector in the column

            index += 1

        return a

    def _build_rhs_vector(self, size):
        """
        Build the right-hand side vector b for the system AX = b, combining
        scalar products of the functions in the system and their derivatives.

        Returns:
            numpy.ndarray: The right-hand side vector b.
        """
        # Initialize the vector b with zeros
        b = np.zeros(size)

        row = 0
        # Iterate through the equations in the system
        for i, callables in enumerate(self.callables_per_equation):
            # Iterate through the parameters in each equation
            for basic_function in callables:
                # Compute the scalar product derivative and accumulate it in b
                b[row] += self._compute_weighted_derivative(i + 1, basic_function, self.ode_results)
                row += 1

        return b

    def solve(self):
        """
        Solve the linear system of equations AX = b.

        Returns:
            numpy.ndarray: Estimated parameters vector.
        """
        # Build normal matrices
        normal_matrices = []
        for basis_functions in self.callables_per_equation:
            normal_matrix = self._compute_normal_matrix(basis_functions, self.ode_results)
            normal_matrices.append(normal_matrix)

        # Build constraints
        r_vectors, c_vectors = self._build_constraints()

        # Build the system matrix A and the vector b
        a = self._build_system_matrix(normal_matrices, r_vectors, c_vectors)
        b = self._build_rhs_vector(np.shape(a)[1])

        # Solve the system
        try:
            solution = self.solver(a, b)
        except Exception as e:
            raise ValueError(f"Error solving the system with solver {self.solver.__name__}. "
                             f"Exception: {str(e)}") from e

        return solution[:self.number_of_parameters]  # Return estimated parameters
