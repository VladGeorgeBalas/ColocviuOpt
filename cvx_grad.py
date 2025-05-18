import cvxpy as cp
import numpy
import numpy as np

def cvx_solve(X, a, b):
    import cvxpy as cp
    import numpy as np

    n, m = X.shape
    Y = cp.Variable((n, m))

    lambda_tv = a
    mu_tv = b

    TV_iso = cp.sum(
        cp.norm(cp.hstack([
            Y[1:, :-1] - Y[:-1, :-1],  # vertical difference
            Y[:-1, 1:] - Y[:-1, :-1]  # horizontal difference
        ]), axis=1)
    )

    TV_aniso = (
            cp.sum(cp.abs(Y[1:, :] - Y[:-1, :])) +
            cp.sum(cp.abs(Y[:, 1:] - Y[:, :-1]))
    )

    objective = cp.Minimize(cp.sum_squares(X - Y) + lambda_tv * TV_iso + mu_tv * TV_aniso)
    constraints = [Y >= 0, Y <= 255]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return Y.value