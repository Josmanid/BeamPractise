import numpy as np
import matplotlib.pyplot as plt

# Validering af FEM for cantilever bjælke ved fri ende
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
    n_dofs = K_global.shape[0]
    f = np.zeros(n_dofs)
    # Påfør en hvis kraft i lodret(nedad) forskydning (u) på noden force_node
    f[2 * force_node] = force_value
    
    # Fastspænding ved node 0: frihedsgrader 0 og 1 er låst, så dem gider vi ikke regne med
    free_dofs = list(range(2, n_dofs))
    
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    f_reduced = f[free_dofs]
    
    # Løs for displaceringer ved frie frihedsgrader
    d_reduced = np.linalg.solve(K_reduced, f_reduced)
    
    d_complete = np.zeros(n_dofs)
    d_complete[free_dofs] = d_reduced
    
    return d_complete

# Sammelinging med analytisk løsning som lært i folkeskolen
def analytical_solution(P, L, E, I, x):
    """
    Analytisk løsning for cantilever bjælke
    """
    return (P * x**2) / (6 * E * I) * (3*L - x)

# Test med forskellige antal elementer
L_total = 1.0  # m
E = 200e9      # Pa
I = 6.67e-10   # m^4
force = -100   # N negativ kraft dvs en nedadgående kraft

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
    d = solve_fem(K, force, n_elem)  # Kraft på sidste node
    
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
analytical_tip = analytical_solution(force, L_total, E, I, L_total) 
print(f"\nAnalytisk løsning ved fri ende: {analytical_tip:.6f} m")

print("\n" + "="*60)
print("KONVERGENS ANALYSE")
print("="*60)

for n_elem in [2, 4, 8]:
    tip_disp = results[n_elem]['tip_displacement']
    error = abs(tip_disp - analytical_tip) / abs(analytical_tip) * 100
    print(f"{n_elem:2d} elementer: u_tip = {tip_disp:.6f} m, Fejl = {error:.2f}%")

# Plot resultater
plt.figure(figsize=(6, 6))

# Plot 1: Deformation sammenligning

x_analytical = np.linspace(0, L_total, 100)
u_analytical = [analytical_solution(force, L_total, E, I, x) for x in x_analytical]


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

# så skal den vises
plt.show()
