import numpy as np

def gaunt_factor_coeffs(n):
    """Johnson (1972) Table 1 Gaunt factor coefficients."""
    if n == 1:
        return 1.1330, -0.4059, 0.07014
    elif n == 2:
        return 1.0785, -0.2319, 0.02947
    else:  # n >= 3
        g0 = 0.9935 + 0.2328 / n - 0.1296 / n**2
        g1 = -(0.6282 - 0.5598 / n + 0.5299 / n**2) / n
        g2 = (0.3887 - 1.181 / n + 1.470 / n**2) / n**2
        return g0, g1, g2

def oscillator_strength(n_lower, n_upper):
    """
    Johnson (1972) Eq. (3) with Gaunt factor Eq. (4).
    Returns f_{n_lower, n_upper} (absorption oscillator strength).
    """
    n = n_lower
    np_ = n_upper
    x = 1.0 - (n / np_)**2  # Eq. (2)
    
    g0, g1, g2 = gaunt_factor_coeffs(n)
    g = g0 + g1 / x + g2 / x**2  # Eq. (4)
    
    prefactor = 32.0 / (3.0 * np.sqrt(3) * np.pi)
    f = prefactor * (n / np_**3) * x**(-3) * g  # Eq. (3)
    return f

Ry = 13.6  # eV, Rydberg energy

def energy_level(n):
    """Hydrogen energy level (binding energy), positive value in eV."""
    return Ry / n**2

def transition_energy(p, n):
    """Energy difference E_pn = E_p - E_n (positive for excitation p < n)."""
    return Ry * (1.0/p**2 - 1.0/n**2)

def A_coeff_vriens(p, n):
    """Vriens & Smeets Eq. (11): A_pn = (2*Ry / E_pn) * f_pn."""
    E_pn = transition_energy(p, n)
    f_pn = oscillator_strength(p, n)
    return (2.0 * Ry / E_pn) * f_pn

def b_p(p):
    """Vriens & Smeets Eq. (13)."""
    return (1.4 * np.log(p)) / p - 0.7 / p - 0.51 / p**2 + 1.16 / p**3 - 0.55 / p**4

def B_coeff_vriens(p, n):
    """
    Vriens & Smeets Eq. (12).
    B_pn = (4 * Ry^2 / n^3) * (1/E_pn^2 + 4*E_pi/(3*E_pn^3) + b_p * E_pi^2 / E_pn^4)
    
    E_pi is the ionization energy of level p.
    E_pn is the excitation energy from p to n.
    """
    E_pn = transition_energy(p, n)
    E_pi = energy_level(p)  # ionization energy of level p = Ry/p^2
    bp = b_p(p)
    
    return (4.0 * Ry**2 / n**3) * (
        1.0 / E_pn**2 
        + 4.0 * E_pi / (3.0 * E_pn**3) 
        + bp * E_pi**2 / E_pn**4
    )

def delta_pn(p, n):
    """Vriens & Smeets Eq. (18)."""
    A = A_coeff_vriens(p, n)
    B = B_coeff_vriens(p, n)
    s = abs(n - p)
    return np.exp(-B / A) + 0.06 * s**2 / (n * p**2)

def gamma_pn(p, n, kTe):
    """
    Vriens & Smeets Eq. (19).
    kTe in eV.
    Returns Gamma_pn in eV.
    """
    s = abs(n - p)
    
    bracket1 = 3.0 + 11.0 * (s / p)**2
    
    bracket2 = (6.0 + 1.6 * n * s 
                + 0.3 / s**2 
                + 0.8 * n**1.5 / s**0.5 * abs(s - 0.6))
    
    return Ry * np.log(1.0 + p**3 * kTe / Ry) * bracket1 / bracket2

def K_excitation(p, n, kTe):
    """
    Vriens & Smeets Eq. (17).
    Excitation rate coefficient for p -> n (n > p).
    kTe in eV.
    Returns K_pn in cm^3 s^-1.
    """
    E_pn = transition_energy(p, n)
    eps_pn = E_pn / kTe          # dimensionless
    A = A_coeff_vriens(p, n)
    B = B_coeff_vriens(p, n)
    d = delta_pn(p, n)
    G = gamma_pn(p, n, kTe)
    
    prefactor = 1.6e-7 * kTe**0.5 / (kTe + G) * np.exp(-eps_pn)
    
    bracket = A * np.log(0.3 * kTe / Ry + d) + B
    
    return prefactor * bracket

def K_ionization(p, kTe):
    """
    Vriens & Smeets Eq. (8).
    Ionization rate coefficient for level p.
    kTe in eV.
    Returns K_pi in cm^3 s^-1.
    """
    E_pi = energy_level(p)       # ionization energy = Ry/p^2
    eps = E_pi / kTe
    
    return (9.56e-6 / (kTe**1.5) * np.exp(-eps) 
            / (eps**2.33 + 4.38 * eps**1.72 + 1.32 * eps))

def einstein_A(n_upper, n_lower):
    f = oscillator_strength(n_lower, n_upper)
    g_lower = 2 * n_lower**2
    g_upper = 2 * n_upper**2
    E_eV = transition_energy(n_lower, n_upper)
    lam_A = 12398.4 / E_eV  # wavelength in Angstroms
    return 6.6703e15 * (g_lower / g_upper) * f / lam_A**2

def K_deexcitation(p, n, kTe):
    """
    De-excitation rate p -> n where p > n.
    From detailed balance: in thermal equilibrium,
    n(n) * ne * K_exc(n->p) = n(p) * ne * K_deex(p->n)
    and n(n)/n(p) = (g_n/g_p) * exp(-E_np/kTe)
    
    So K_deex(p->n) = (g_n/g_p) * K_exc(n->p) * exp(+E_np/kTe)
    
    kTe in eV. Returns cm^3 s^-1.
    """
    g_n = 2 * n**2
    g_p = 2 * p**2
    E = transition_energy(n, p)  # positive, energy gap
    return (g_n / g_p) * K_excitation(n, p, kTe) * np.exp(E / kTe)

def K_recombination(p, kTe):
    """
    Vriens & Smeets Eq. (9). Three-body recombination into level p.
    Rate per unit volume = ne^2 * alpha, so alpha has units cm^6 s^-1.
    """
    E_pi = energy_level(p)
    eps = E_pi / kTe
    g_p = 2 * p**2
    g_ion = 1  # proton ground state
    
    return (3.17e-27 * kTe**(-1.5) * (g_p / g_ion) 
            / (eps**2.33 + 4.38 * eps**1.72 + 1.32 * eps))

def build_rate_matrix(n_levels, kTe, ne):
    """
    Build the steady-state rate matrix M where M @ n_pop = source.
    
    n_levels: number of levels (e.g. 15 means n=1 to n=15)
    kTe: electron temperature in eV
    ne: electron density in cm^-3
    
    We fix n(1) = 1 (ground state population) and solve for n(2)...n(n_levels).
    The matrix equation is (n_levels - 1) x (n_levels - 1).
    """
    N = n_levels
    
    # indices: level n corresponds to array index n-2 (since we exclude n=1)
    # so level 2 -> index 0, level 3 -> index 1, etc.
    
    size = N - 1  # number of unknowns: levels 2 through N
    M = np.zeros((size, size))
    source = np.zeros(size)
    
    for i_lvl in range(2, N + 1):  # level i
        row = i_lvl - 2  # row index in matrix
        
        # === terms that REMOVE population from level i (go on diagonal) ===
        loss = 0.0
        
        # excitation from i to higher levels
        for j in range(i_lvl + 1, N + 1):
            loss += ne * K_excitation(i_lvl, j, kTe)
        
        # de-excitation from i to lower levels
        for j in range(1, i_lvl):
            loss += ne * K_deexcitation(i_lvl, j, kTe)
        
        # radiative decay from i to lower levels
        for j in range(1, i_lvl):
            loss += einstein_A(i_lvl, j)
        
        # ionization from i
        loss += ne * K_ionization(i_lvl, kTe)
        
        M[row, row] = -loss
        
        # === terms that ADD population to level i (off-diagonal and source) ===
        
        for j in range(2, N + 1):  # from level j (j != i, j >= 2)
            if j == i_lvl:
                continue
            col = j - 2  # column index
            
            if j < i_lvl:
                # excitation from j up to i
                M[row, col] += ne * K_excitation(j, i_lvl, kTe)
            else:
                # de-excitation from j down to i
                M[row, col] += ne * K_deexcitation(j, i_lvl, kTe)
                # radiative decay from j down to i
                M[row, col] += einstein_A(j, i_lvl)
        
        # === contributions from level 1 (ground state, fixed n(1)=1) -> source ===
        # excitation from ground state to level i
        source[row] -= ne * K_excitation(1, i_lvl, kTe) * 1.0  # n(1) = 1
        # recombination from continuum into level i
        #source[row] -= ne**2 * K_recombination(i_lvl, kTe)
        
    return M, source


def solve_steady_state(n_levels, kTe, ne):
    """
    Solve for level populations. Returns array of populations n(1)...n(n_levels).
    n(1) is fixed to 1, everything else is relative to that.
    """
    M, source = build_rate_matrix(n_levels, kTe, ne)
    
    # solve M @ x = source for x = [n(2), n(3), ..., n(N)]
    pops_excited = np.linalg.solve(M, source)
    
    # assemble full population array
    pops = np.zeros(n_levels)
    pops[0] = 1.0  # n(1) = 1
    pops[1:] = pops_excited
    
    return pops


def balmer_ratio(pops, ratio="ba"):
    """
    Compute Balmer line ratios from populations.
    Intensity of line is proportional to A(upper->2) * n(upper).
    
    ratio="ba" -> Hbeta/Halpha = [A(4->2)*n(4)] / [A(3->2)*n(3)]
    ratio="ga" -> Hgamma/Halpha
    ratio="gb" -> Hgamma/Hbeta
    """
    I_alpha = einstein_A(3, 2) * pops[2]  # n=3, index 2
    I_beta  = einstein_A(4, 2) * pops[3]  # n=4, index 3
    I_gamma = einstein_A(5, 2) * pops[4]  # n=5, index 4
    
    if ratio == "ba":
        return I_beta / I_alpha
    elif ratio == "ga":
        return I_gamma / I_alpha
    elif ratio == "gb":
        return I_gamma / I_beta
def compute_ratio_grid(n_levels, kTe_arr, ne_arr, ratio_type="ba"):
    """
    Compute Balmer line ratio over a 2D grid of (kTe, ne).
    Returns 2D array of shape (len(ne_arr), len(kTe_arr)).
    """
    result = np.zeros((len(ne_arr), len(kTe_arr)))
    
    for i, ne in enumerate(ne_arr):
        for j, kTe in enumerate(kTe_arr):
            pops = solve_steady_state(n_levels, kTe, ne)
            result[i, j] = balmer_ratio(pops, ratio_type)
        print(f"  ne = {ne:.1e} done")
    
    return result

def test_oscillator_strengths():
    print("=== Oscillator Strengths ===")
    print("f(1,2) =", oscillator_strength(1, 2))
    print("f(1,3) =", oscillator_strength(1, 3))
    print("f(2,3) =", oscillator_strength(2, 3))
    print("f(2,4) =", oscillator_strength(2, 4))
    print("f(3,4) =", oscillator_strength(3, 4))
    print("f(3,5) =", oscillator_strength(3, 5))

def test_bethe_coefficients():
    print("=== Bethe Coefficients ===")
    for p, n in [(1,2), (2,3), (5,6), (5,8), (10,11)]:
        A = A_coeff_vriens(p, n)
        B = B_coeff_vriens(p, n)
        print(f"p={p}, n={n}: A_pn={A:.4f}, B_pn={B:.4f}, B/A={B/A:.4f}")

def test_excitation_rates():
    print("=== Excitation Rates ===")
    for kTe in [1.0, 5.0, 10.0, 50.0, 100.0]:
        K_12 = K_excitation(1, 2, kTe)
        K_23 = K_excitation(2, 3, kTe)
        K_56 = K_excitation(5, 6, kTe)
        print(f"kTe={kTe:6.1f} eV: K(1->2)={K_12:.3e}, K(2->3)={K_23:.3e}, K(5->6)={K_56:.3e}")

def test_ionization_and_einstein():
    print("=== Ionization Rates ===")
    for kTe in [1.0, 10.0, 100.0]:
        print(f"kTe={kTe:.0f} eV: K_ion(1)={K_ionization(1, kTe):.3e}, "
              f"K_ion(3)={K_ionization(3, kTe):.3e}, "
              f"K_ion(5)={K_ionization(5, kTe):.3e}")
    print()
    print("=== Einstein A Coefficients ===")
    print(f"A(2->1) = {einstein_A(2,1):.3e}  ref: 4.697e8 (Berestetskii)")
    print(f"A(3->2) = {einstein_A(3,2):.3e}  ref: 4.408e7 (Berestetskii)")
    print(f"A(3->1) = {einstein_A(3,1):.3e}  ref: 5.573e7 (Berestetskii)")
    print(f"A(4->2) = {einstein_A(4,2):.3e}  ref: 8.416e6 (Berestetskii)")

def test_deexcitation():
    print("=== De-excitation Rates ===")
    kTe = 5.0
    for p, n in [(3, 2), (4, 2), (6, 5)]:
        K_ex = K_excitation(n, p, kTe)
        K_de = K_deexcitation(p, n, kTe)
        print(f"kTe={kTe} eV: K_exc({n}->{p})={K_ex:.3e}, "
              f"K_deex({p}->{n})={K_de:.3e}, ratio={K_de/K_ex:.2f}")

def test_convergence():
    print("=== Convergence Test ===")
    kTe = 10.0
    ne = 1e12
    for N in [10, 12, 15, 18]:
        pops = solve_steady_state(N, kTe, ne)
        ratio_ba = balmer_ratio(pops, 'ba')
        ratio_gb = balmer_ratio(pops, 'gb')
        print(f"N={N:2d}: Hb/Ha={ratio_ba:.4f}, Hg/Hb={ratio_gb:.4f}, "
              f"n(3)={pops[2]:.3e}, n(last)={pops[-1]:.3e}")

def test_limits():
    print("Sanity Checks Coronal ")
    kTe = 10.0
    
    # coronal limit
    A_sum_3 = einstein_A(3,1) + einstein_A(3,2)
    A_sum_4 = einstein_A(4,1) + einstein_A(4,2) + einstein_A(4,3)
    n3_coronal = K_excitation(1, 3, kTe) / A_sum_3
    n4_coronal = K_excitation(1, 4, kTe) / A_sum_4
    ratio_coronal = (einstein_A(4, 2) * n4_coronal) / (einstein_A(3, 2) * n3_coronal)
    
    pops_low = solve_steady_state(15, kTe, 1e8)
    ratio_model = balmer_ratio(pops_low, "ba")
    
    print(f"Coronal limit (analytic): Hb/Ha = {ratio_coronal:.4f}")
    print(f"Full model at ne=1e8:     Hb/Ha = {ratio_model:.4f}")
    
    # LTE limit
    # E_exc_3 = transition_energy(1, 3)
    # E_exc_4 = transition_energy(1, 4)
    # n3_boltz = 9.0 * np.exp(-E_exc_3 / kTe)
    # n4_boltz = 16.0 * np.exp(-E_exc_4 / kTe)
    # ratio_lte = (einstein_A(4, 2) * n4_boltz) / (einstein_A(3, 2) * n3_boltz)
    
    # pops_very_high = solve_steady_state(15, kTe, 1e21)
    # ratio_very_high = balmer_ratio(pops_very_high, "ba")
    
    # print(f"\nLTE limit (Boltzmann):    Hb/Ha = {ratio_lte:.4f}")
    # print(f"Full model at ne=1e21:   Hb/Ha = {ratio_very_high:.4f}")
    # print(f"\nBoltzmann check at ne=1e21, kTe={kTe}:")
    # for i in [2, 3, 4, 5]:
    #     E_exc = transition_energy(1, i)
    #     boltz = i**2 * np.exp(-E_exc / kTe)
    #     print(f"  n({i})/n(1): model={pops_very_high[i-1]:.4e}, Boltzmann={boltz:.4e}, "
    #           f"ratio={pops_very_high[i-1]/boltz:.4f}")

def run_contour_plots():
    import matplotlib.pyplot as plt
    
    print("=== Computing Contour Plots ===")
    n_levels = 15
    kTe_arr = np.logspace(0, 2.3, 30)
    ne_arr = np.logspace(10, 15, 30)
    
    print("Computing Hbeta/Halpha grid...")
    ratio_ba = compute_ratio_grid(n_levels, kTe_arr, ne_arr, "ba")
    
    print("Computing Hgamma/Hbeta grid...")
    ratio_gb = compute_ratio_grid(n_levels, kTe_arr, ne_arr, "gb")
    
    Te_grid, Ne_grid = np.meshgrid(kTe_arr, ne_arr)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    c1 = ax1.contourf(Te_grid, Ne_grid, ratio_ba, levels=20, cmap='viridis')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('kTe (eV)')
    ax1.set_ylabel('ne (cm^-3)')
    ax1.set_title('Hbeta / Halpha')
    plt.colorbar(c1, ax=ax1)
    
    c2 = ax2.contourf(Te_grid, Ne_grid, ratio_gb, levels=20, cmap='viridis')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('kTe (eV)')
    ax2.set_ylabel('ne (cm^-3)')
    ax2.set_title('Hgamma / Hbeta')
    plt.colorbar(c2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('balmer_ratios.png', dpi=150)
    plt.show()
    print("Done!")
    import matplotlib.pyplot as plt
    
    print("=== Computing Contour Plots ===")
    n_levels = 15
    kTe_arr = np.linspace(0, 2.3, 30)
    ne_arr = np.linspace(10, 15, 30)
    
    print("Computing Hbeta/Halpha grid...")
    ratio_ba = compute_ratio_grid(n_levels, kTe_arr, ne_arr, "ba")
    
    print("Computing Hgamma/Hbeta grid...")
    ratio_gb = compute_ratio_grid(n_levels, kTe_arr, ne_arr, "gb")
    
    Te_grid, Ne_grid = np.meshgrid(kTe_arr, ne_arr)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    c1 = ax1.contourf(Te_grid, Ne_grid, ratio_ba, levels=20, cmap='viridis')
    ax1.set_xlabel('kTe (eV)')
    ax1.set_ylabel('ne (cm^-3)')
    ax1.set_title('Hbeta / Halpha')
    plt.colorbar(c1, ax=ax1)
    
    c2 = ax2.contourf(Te_grid, Ne_grid, ratio_gb, levels=20, cmap='viridis')
    ax2.set_xlabel('kTe (eV)')
    ax2.set_ylabel('ne (cm^-3)')
    ax2.set_title('Hgamma / Hbeta')
    plt.colorbar(c2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('balmer_ratios.png', dpi=150)
    plt.show()
    print("Done!")


if __name__ == "__main__":
    # comment/uncomment what you want to run
    #test_oscillator_strengths()
    #test_bethe_coefficients()
    #test_excitation_rates()
   # test_ionization_and_einstein()
    #test_deexcitation()
    #test_convergence()
    #test_limits()
    run_contour_plots()