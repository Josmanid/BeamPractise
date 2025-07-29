import numpy as np

# hele Stivhedsmatrixen
K = np.array([
    [12806.4, 3201.6, -12806.4, 3201.6, 0, 0],
    [3201.6, 1067.2, -3201.6, 533.6, 0, 0],
    [-12806.4, -3201.6, 25612.8, 0, -12806.4, 3201.6],
    [3201.6, 533.6, 0, 2134.4, -3201.6, 533.6],
    [0, 0, -12806.4, -3201.6, 12806.4, -3201.6],
    [0, 0, 3201.6, 533.6, -3201.6, 1067.2]
])

# Kraf vector 
F = np.array([0, 0, -100, 0, 0, 0])  # N

# Node 0 er fastspændt, så vi fjerner frihedsgraderne d0 og d1
# Derfor fjerner vi de første 2 rækker og kolonner fra K og de første 2 elementer fra F
F_reduced = F[2:]
K_reduced = K[2:, 2:]



# Løsning af systemet K * u = F
d_reduced = np.linalg.solve(K_reduced, F_reduced)
# Saml hele løsningen med kendte værdier
d = np.zeros(6)
d[2:] = d_reduced

print("Forskydninger og rotationer (d):")
print(d)