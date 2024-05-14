"""
Júlia Pivato de Oliveira
Mirela Vitória Domiciano

Método de Gauss-Seidel
"""

def gauss_seidel(A, b, x0, tol=1e-6, max_iter=1000):
    n = len(b)
    x = x0[:]
    x_new = x0[:]
    iter_count = 0

    while iter_count < max_iter:
        for i in range(n):
            sum_ax = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x[i] = (b[i] - sum_ax) / A[i][i]

        if all(abs(x_new[i] - x[i]) < tol for i in range(n)):
            return x

        x_new = x[:]
        iter_count += 1

    raise ValueError("O método de Gauss-Seidel não convergiu após o número máximo de iterações.")

# Coeficientes da matriz A
A = [
    [5,  2, -1,  1],
    [2,  6,  2, -1],
    [1,  2,  7,  3],
    [3, -1,  2,  8]
]

# Termos independentes
b = [12, 10, 17, 11]

# Valor inicial para x
x0 = [0.0] * len(b)

# Chamar a função para resolver o sistema
x_solution = gauss_seidel(A, b, x0)
print("Solução = [", end="")
for i, value in enumerate(x_solution):
    if i != len(x_solution) - 1:
        print(f"{value:.8f}, ", end="")
    else:
        print(f"{value:.8f}]", end="")
