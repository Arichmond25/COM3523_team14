"""
The ``exceptions`` module houses custom exceptions. Currently implemented:

- OptimizationError
"""

class OptimizationError(Exception):
    """
    Raised when an optimization routine fails – typically when
    cvxpy does not return an 'optimal' flag.

    Args:
        message (str): Custom error message.
        solver (str, optional): Name of the solver used.
        status (str, optional): Solver return status.
    """

    def __init__(self, message=None, solver=None, status=None):
        self.message = message or "Optimization failed. Please check your objectives/constraints or try a different solver."
        self.solver = solver
        self.status = status
        super().__init__(self.message)

    def __str__(self):
        info = self.message
        if self.solver:
            info += f" | Solver: {self.solver}"
        if self.status:
            info += f" | Status: {self.status}"
        return info
