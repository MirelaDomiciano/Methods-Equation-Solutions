"""
Júlia Pivato de Oliveira
Mirela Vitória Domiciano

Eliminação de Gauss
"""
import numpy as np

matrix = np.array([
    [2, 2, 1, 1, 7],
    [1, -1, 2, -1, 1],
    [3, 2, -3, -2, 4],
    [4, 3, 2, 1, 12]
], dtype=float)
A = np.array([
    [2, 2, 1, 1],
    [1, -1, 2, -1],
    [3, 2, -3, -2],
    [4, 3, 2, 1]
], dtype=float)

def det(matrix):
    if(matrix.shape[0] == matrix.shape[1]):
        width = len(matrix)
        indices = np.array(range(width))
        total = 0  

        if width == 2 and len(matrix[0] == 2):
            total = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        else:
            for fc in indices: 
                m = np.array([row[:fc] + row[fc+1:] for row in (matrix[1:])])
                s = (-1) ** (fc % 2)
                sub_det = det(m)
                total += s * matrix[0][fc] * sub_det
        if total != 0:
            return True
    return False


def gauss(matrix):
    if np.isclose(det(matrix), 0):
        n = matrix.shape[0]

        for i in range(n):
            # Escolha do pivo
            max_row = i
            for j in range(i+1, n):
                if abs(matrix[j, i]) > abs(matrix[max_row, i]):
                    max_row = j
            matrix[[i, max_row]] = matrix[[max_row, i]]

            #  Substituição da linha por  L2 – (a21/a11) *L1
            for j in range(i+1, n):
                factor = matrix[j, i] / matrix[i, i]
                matrix[j, i:] -= factor * matrix[i, i:]
        matrix = np.round(matrix, 2)
        print(np.array2string(matrix, formatter={'float_kind': lambda x: "%.1f" % x}), "\n")
        # Solução do sistema
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = (matrix[i, -1] - matrix[i, i+1:n] @ x[i+1:]) / matrix[i, i]
        return x
    else:
        print("Determinante igual a 0!")
        return None
np.set_printoptions(precision=2)
sol = gauss(matrix)
sol = np.where(np.abs(sol) <= -0.0, 0.0, sol)
print("Solução:", np.array2string(sol, formatter={'float_kind': lambda x: "%.1f" % x}))
