import numpy as np

def f(x):
    x1, x2 = x
    return np.array([
        4*x1 - x1**3 + x2,
        -x1**2/9 + (4*x2 - x2**2)/4 + 1
    ])

def jacobian(x):
    x1, x2 = x
    return np.array([
        [4 - 3*x1**2, 1],
        [-2*x1/9, (4 - 2*x2)/4]
    ])

def newton_method(f, jacobian, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        J = jacobian(x)
        f_val = f(x)
        delta_x = np.linalg.solve(J, -f_val)
        x = x + delta_x
        if np.linalg.norm(delta_x) < tol:
            return x
    raise ValueError('Newton method did not converge')

# Chute inicial
x0 = np.array([-1, -2])

solution = newton_method(f, jacobian, x0)
solution_rounded = np.round(solution, 5)
print('Solution:', solution_rounded)
