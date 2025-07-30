import numpy as np

# Materiale- og geometriparametre
# Her antager vi, at bjælken er lavet af stål
# E er Youngs modul, I er det tværgående inertimoment, og Le er elementlængden
E = 200e9        # Pa
I = 6.67e-10     # m^4
Le = 0.5         # m

# Konstantfaktor
factor = (E * I) / (Le ** 3)  # N/m
print("Konstantfaktor:", factor)

# Lokal stivhedsmatrix (uden EI/L^3 faktoren)
k_local_base = np.array([
    [12,     3,   -12,     3],
    [3,      1,    -3,   0.5],
    [-12,   -3,    12,    -3],
    [3,    0.5,    -3,     1]
])

# Skaleret lokal matrix
k_local = factor * k_local_base

# Initialiser global 6x6 matrix
K_global = np.zeros((6, 6))

# Element 1 indsættes ved DOFs [0,1,2,3]
dofs_elem1 = [0, 1, 2, 3]
for i in range(4):
    for j in range(4):
        K_global[dofs_elem1[i], dofs_elem1[j]] += k_local[i, j]

# Element 2 indsættes ved DOFs [2,3,4,5]
dofs_elem2 = [2, 3, 4, 5]
for i in range(4):
    for j in range(4):
        K_global[dofs_elem2[i], dofs_elem2[j]] += k_local[i, j]

# Udskriv global matrix
np.set_printoptions(precision=1, suppress=True)
print("\nGlobal stivhedsmatrix K:")
print(K_global)
