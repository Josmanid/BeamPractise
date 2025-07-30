import numpy as np
import matplotlib.pyplot as plt

def create_global_stiffness_matrix(n_elements, L_total, E, I):
    """
    Opretter global stivhedsmatrix for cantilever bjælke med n_elements
    
    Parameters:
    n_elements: antal elementer
    L_total: total længde af bjælke
    E: Young's modulus
    I: Inertimoment
    """
    
    # Beregn elementlængde
    Le = L_total / n_elements
    
    # Antal noder og DOFs
    n_nodes = n_elements + 1
    n_dofs = n_nodes * 2  # 2 DOFs per node (u, theta)
    
    print(f"Antal elementer: {n_elements}")
    print(f"Antal noder: {n_nodes}")
    print(f"Elementlængde: {Le:.3f} m")
    print(f"Total DOFs: {n_dofs}")
    
    # Konstantfaktor
    factor = (E * I) / (Le ** 3)
    
    # Lokal stivhedsmatrix (normaliseret)
    k_local_base = np.array([
        [12,    6*Le,   -12,    6*Le],
        [6*Le,  4*Le**2, -6*Le,  2*Le**2],
        [-12,   -6*Le,   12,    -6*Le],
        [6*Le,  2*Le**2, -6*Le,  4*Le**2]
    ])
    
    # Skaleret lokal matrix
    k_local = factor * k_local_base
    
    # Initialiser global matrix
    K_global = np.zeros((n_dofs, n_dofs))
    
    # Samling af elementer
    for elem in range(n_elements):
        # DOFs for dette element (node i og i+1)
        node_i = elem
        node_j = elem + 1
        
        # DOF indices: hver node har 2 DOFs (u, theta)
        dofs = [2*node_i, 2*node_i+1, 2*node_j, 2*node_j+1]
        
        # Indsæt lokal matrix i global matrix
        for i in range(4):
            for j in range(4):
                K_global[dofs[i], dofs[j]] += k_local[i, j]
    
    return K_global, Le, factor

def solve_fem(K_global, force_value, force_node):
    """
    Løser FEM systemet med randbetingelser
    """
    n_dofs = K_global.shape[0]
    
    # Kraftvektor - kraft på den sidste node (fri ende)
    f = np.zeros(n_dofs)
    last_node = (n_dofs // 2) - 1  # Sidste node index
    f[2*last_node] = force_value  # Lodret kraft på fri ende
    
    # Randbetingelser: node 0 er fastspændt (u=0, theta=0)
    free_dofs = list(range(2, n_dofs))  # Alle DOFs undtagen de første to
    
    # Reduceret system
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    f_reduced = f[free_dofs]
    
    # Løs systemet
    d_reduced = np.linalg.solve(K_reduced, f_reduced)
    
    # Komplet forskydningsvektor
    d_complete = np.zeros(n_dofs)
    d_complete[free_dofs] = d_reduced
    
    return d_complete

def analytical_solution(P, L, E, I, x):
    """
    Analytisk løsning for cantilever bjælke
    """
    return (P * x**2) / (6 * E * I) * (3*L - x)

# Test med forskellige antal elementer
L_total = 1.0  # m
E = 200e9      # Pa
I = 6.67e-10   # m^4
force = -100   # N

print("="*60)
print("SAMMENLIGNING AF FORSKELLIGE ELEMENT-ANTAL")
print("="*60)

results = {}

for n_elem in [2, 4, 8]:
    print(f"\n--- {n_elem} ELEMENTER ---")
    
    # Opret global matrix
    K, Le, factor = create_global_stiffness_matrix(n_elem, L_total, E, I)
    
    print(f"\nGlobal stivhedsmatrix shape: {K.shape}")
    if n_elem == 2:
        print("Global matrix (kun for 2 elementer):")
        np.set_printoptions(precision=1, suppress=True)
        print(K)
    
    # Løs systemet (kraft på fri ende - sidste node)
    d = solve_fem(K, force, 0)  # Parameter bruges ikke længere
    
    # Udtræk forskydninger ved noderne
    nodes_x = np.linspace(0, L_total, n_elem + 1)
    displacements = d[::2]  # Hver anden værdi (kun u, ikke theta)
    
    # Gem resultater
    results[n_elem] = {
        'nodes_x': nodes_x,
        'displacements': displacements,
        'tip_displacement': displacements[-1]
    }
    
    print(f"Forskydning ved fri ende: {displacements[-1]:.6f} m")

# Analytisk løsning ved fri ende
analytical_tip = analytical_solution(abs(force), L_total, E, I, L_total)
print(f"\nAnalytisk løsning ved fri ende: {analytical_tip:.6f} m")

print("\n" + "="*60)
print("KONVERGENS ANALYSE")
print("="*60)

for n_elem in [2, 4, 8]:
    tip_disp = results[n_elem]['tip_displacement']
    error = abs(tip_disp - analytical_tip) / analytical_tip * 100
    print(f"{n_elem:2d} elementer: u_tip = {tip_disp:.6f} m, Fejl = {error:.2f}%")

# Plot resultater
plt.figure(figsize=(12, 8))

# Plot 1: Deformation comparison
plt.subplot(2, 2, 1)
x_analytical = np.linspace(0, L_total, 100)
u_analytical = [analytical_solution(abs(force), L_total, E, I, x) for x in x_analytical]
u_analytical = [-u for u in u_analytical]  # Negativ for nedad retning

plt.plot(x_analytical, u_analytical, 'k-', linewidth=2, label='Analytisk')

for n_elem in [2, 4, 8]:
    data = results[n_elem]
    plt.plot(data['nodes_x'], data['displacements'], 'o-', 
             label=f'{n_elem} elementer', markersize=6)

plt.xlabel('Position (m)')
plt.ylabel('Forskydning (m)')
plt.title('Deformation af cantilever bjælke')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Konvergens
plt.subplot(2, 2, 2)
n_elements_list = [2, 4, 8, 16, 32]
tip_displacements = []

for n_elem in n_elements_list:
    if n_elem in results:
        tip_displacements.append(abs(results[n_elem]['tip_displacement']))
    else:
        # Beregn for flere elementer
        K, _, _ = create_global_stiffness_matrix(n_elem, L_total, E, I)
        d = solve_fem(K, force, 0)  # Parameter bruges ikke
        tip_displacements.append(abs(d[-2]))  # Sidste node, u DOF

plt.semilogx(n_elements_list, tip_displacements, 'bo-', label='FEM')
plt.axhline(y=analytical_tip, color='red', linestyle='--', label='Analytisk')
plt.xlabel('Antal elementer')
plt.ylabel('Forskydning ved fri ende (m)')
plt.title('Konvergens studie')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Matrix sparsity pattern for 8 elementer
plt.subplot(2, 2, 3)
K_8, _, _ = create_global_stiffness_matrix(8, L_total, E, I)
plt.spy(K_8, markersize=3)
plt.title('Stivhedsmatrix struktur (8 elementer)')
plt.xlabel('Column')
plt.ylabel('Row')

# Plot 4: Error vs elements
plt.subplot(2, 2, 4)
errors = []
for i, n_elem in enumerate(n_elements_list):
    error = abs(tip_displacements[i] - analytical_tip) / analytical_tip * 100
    errors.append(error)

plt.loglog(n_elements_list, errors, 'ro-')
plt.xlabel('Antal elementer')
plt.ylabel('Relativ fejl (%)')
plt.title('Fejl vs. element antal')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nMatrix dimensioner:")
for n_elem in [2, 4, 8]:
    n_dofs = (n_elem + 1) * 2
    print(f"{n_elem} elementer: {n_dofs}×{n_dofs} matrix")