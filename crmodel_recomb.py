import numpy as np
from scipy.integrate import solve_ivp

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
    """the energy level"""
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
    Rate per unit volume = ne^2 * alpha, alpha has units cm^6 s^-1!!!!
    """
    E_pi = energy_level(p)
    eps = E_pi / kTe
    g_p = 2 * p**2
    g_ion = 1  # proton ground state
    
    return (3.17e-27 * kTe**(-1.5) * (g_p / g_ion) 
            / (eps**2.33 + 4.38 * eps**1.72 + 1.32 * eps))

def build_rate_matrix(n_levels, kTe, ne, use_recomb=True, rate_scale=1.0):
    """
    Build the steady-state rate matrix M where M @ n_pop = source.
    
    n_levels: number of levels (e.g. 15 means n=1 to n=15)
    kTe: electron temperature in eV
    ne: electron density in cm^-3
    
    Fix n(1) = 1 (ground state population) and solve for n(2)...n(n_levels).
    The matrix equation is (n_levels - 1) x (n_levels - 1).
    """
    N = n_levels
    
    # indices: level n corresponds to array index n-2
    
    size = N - 1  # number of unknowns: levels 2 through N
    M = np.zeros((size, size))
    source = np.zeros(size)
    
    for i_lvl in range(2, N + 1):  # level i
        row = i_lvl - 2  # row index in matrix
        
        # =terms that REMOVE population from level i (go on diagonal)
        loss = 0.0
        
        # excitation from i to higher levels
        for j in range(i_lvl + 1, N + 1):
            loss += ne * K_excitation(i_lvl, j, kTe) * rate_scale
        
        # de-excitation from i to lower levels
        for j in range(1, i_lvl):
            loss += ne * K_deexcitation(i_lvl, j, kTe) * rate_scale
        
        # radiative decay from i to lower levels
        for j in range(1, i_lvl):
            loss += einstein_A(i_lvl, j)
        
        # ionization from i
        loss += ne * K_ionization(i_lvl, kTe)
        
        M[row, row] = -loss
        
        # terms that ADD population to level i (off-diagonal and source) 
        
        for j in range(2, N + 1):  # from level j (j != i, j >= 2)
            if j == i_lvl:
                continue
            col = j - 2  # column index
            
            if j < i_lvl:
                # excitation from j up to i
                M[row, col] += ne * K_excitation(j, i_lvl, kTe) * rate_scale
            else:
                # de-excitation from j down to i
                M[row, col] += ne * K_deexcitation(j, i_lvl, kTe) * rate_scale
                # radiative decay from j down to i
                M[row, col] += einstein_A(j, i_lvl)
        
        # contributions from level 1 (ground state, fixed n(1)=1) -> source
        # excitation from ground state to level i
        source[row] -= ne * rate_scale * K_excitation(1, i_lvl, kTe)  # n(1) = 1
        # recombination from continuum into level i
        if use_recomb:
            source[row] -= ne**2 * rate_scale * K_recombination(i_lvl, kTe)
        
    return M, source


def solve_steady_state(n_levels, kTe, ne, use_recomb=True, rate_scale=1.0):
    M, source = build_rate_matrix(n_levels, kTe, ne, use_recomb, rate_scale)
    pops_excited = np.linalg.solve(M, source)
    pops = np.zeros(n_levels)
    pops[0] = 1.0
    pops[1:] = pops_excited
    return pops


def balmer_ratio(pops, ratio="ba"):
    """
    Compute Balmer line ratios from pops
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

def test_coronal_limit():
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

def test_lte():
    kTe = 10.0
    E_exc_3 = transition_energy(1, 3)
    E_exc_4 = transition_energy(1, 4)
    n3_boltz = 9.0 * np.exp(-E_exc_3 / kTe)
    n4_boltz = 16.0 * np.exp(-E_exc_4 / kTe)
    ratio_lte = (einstein_A(4, 2) * n4_boltz) / (einstein_A(3, 2) * n3_boltz)
    
    pops = solve_steady_state(15, kTe, 1e21)
    ratio_model = balmer_ratio(pops, "ba")
    
    print(f"LTE (Boltzmann):  Hb/Ha = {ratio_lte:.4f}")
    print(f"Model at ne=1e21: Hb/Ha = {ratio_model:.4f}")

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

def plot_vinoth_comparison():
    import matplotlib.pyplot as plt
    
    n_levels = 15
    ne_arr = np.logspace(10, 15, 50)
    temps = [10, 80, 100, 200, 400]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    for kTe, color in zip(temps, colors):
        ratios_ba = []
        ratios_gb = []
        for ne in ne_arr:
            pops = solve_steady_state(n_levels, kTe, ne)
            ratios_ba.append(balmer_ratio(pops, 'ba'))
            ratios_gb.append(balmer_ratio(pops, 'gb'))
        ax1.plot(ne_arr, ratios_ba, color=color, label=f'{kTe} eV')
        ax2.plot(ne_arr, ratios_gb, color=color, label=f'{kTe} eV')
        print(f"kTe = {kTe} eV done")
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1e10, 1e15)
    ax1.set_ylim(0.06, 0.25)
    ax1.set_xlabel(r'Electron density (cm$^{-3}$)')
    ax1.set_ylabel(r'H$_\beta$/H$_\alpha$')
    ax1.legend()
    ax1.set_title('(a)')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1e10, 1e15)
    ax2.set_ylim(0.1, 0.4)
    ax2.set_xlabel(r'Electron density (cm$^{-3}$)')
    ax2.set_ylabel(r'H$_\gamma$/H$_\beta$')
    ax2.legend()
    ax2.set_title('(b)')
    
    plt.tight_layout()
    plt.savefig('vinoth_comparison.png', dpi=150)
    plt.show()

def run_equilibration_test(n_levels, kTe_init, kTe_final, ne):
    """
    Start at steady state for kTe_init switch to kTe_final
    """
    pops_init = solve_steady_state(n_levels, kTe_init, ne)
    pops_final = solve_steady_state(n_levels, kTe_final, ne)
    
    y0 = pops_init[1:]  # excited levels only
    
    M, source = build_rate_matrix(n_levels, kTe_final, ne)
    
    def rhs(t, y):
        return M @ y - source
    
    t_end = 1e-5  # 10 microseconds
    
    sol = solve_ivp(rhs, [0, t_end], y0, method='BDF',
                    rtol=1e-8, atol=1e-12, dense_output=True)
    
    return sol, pops_init, pops_final

def plot_equilibration():
    import matplotlib.pyplot as plt
    
    n_levels = 15
    ne = 1e12
    kTe_init = 10.0
    kTe_final = 50.0
    
    sol, pops_init, pops_final = run_equilibration_test(
        n_levels, kTe_init, kTe_final, ne)
    
    t = sol.t
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # plot level populations vs time
    for lvl in [2, 3, 4, 5]:
        idx = lvl - 2
        y = sol.y[idx]
        y_norm = (y - pops_init[lvl-1]) / (pops_final[lvl-1] - pops_init[lvl-1])
        ax1.plot(t, y_norm, label=f'n={lvl}')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Normalized population')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.set_title(f'Level equilibration: $kT_e$ {kTe_init} → {kTe_final} eV, '
                  f'$n_e$ = {ne:.0e} cm$^{{-3}}$')
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    
    # plot Balmer ratio vs time
    t_dense = np.logspace(np.log10(max(t[0], 1e-15)), np.log10(t[-1]), 200)
    ratios = []
    for ti in t_dense:
        y = sol.sol(ti)
        full_pops = np.zeros(n_levels)
        full_pops[0] = 1.0
        full_pops[1:] = y
        ratios.append(balmer_ratio(full_pops, 'ba'))
    
    ax2.plot(t_dense, ratios)
    ax2.axhline(balmer_ratio(pops_final, 'ba'), color='red',
                linestyle='--', label='New steady state')
    ax2.axhline(balmer_ratio(pops_init, 'ba'), color='blue',
                linestyle='--', label='Old steady state')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r'H$\beta$/H$\alpha$')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.set_title(r'H$\beta$/H$\alpha$ relaxation')
    
    plt.tight_layout()
    plt.savefig('equilibration.png', dpi=150)
    plt.show()

def equilibration_timescales(n_levels, kTe, ne):

    M, source = build_rate_matrix(n_levels, kTe, ne)
    eigenvalues = np.linalg.eigvals(M)
    
    # eigenvalues should all be negative (stable system)
    # timescales are -1/eigenvalue
    timescales = -1.0 / eigenvalues.real
    timescales = np.sort(timescales)
    
    return timescales

def run_step_response_with_eigenvalues():
   
    import matplotlib.pyplot as plt
    
    n_levels = 15
    kTe_init = 10.0
    kTe_final = 50.0
    
    densities = [1e10, 1e12, 1e14]
    
    fig, axes = plt.subplots(len(densities), 1, figsize=(10, 4*len(densities)))
    
    for ax, ne in zip(axes, densities):
        # BDF integration
        sol, pops_init, pops_final = run_equilibration_test(
            n_levels, kTe_init, kTe_final, ne)
        
        # eigenvalue prediction
        ts = equilibration_timescales(n_levels, kTe_final, ne)
        tau_slowest = ts[-1]
        
        # plot Balmer ratio vs time
        t_dense = np.logspace(np.log10(max(sol.t[0], 1e-15)), 
                              np.log10(sol.t[-1]), 300)
        ratios = []
        for ti in t_dense:
            y = sol.sol(ti)
            full_pops = np.zeros(n_levels)
            full_pops[0] = 1.0
            full_pops[1:] = y
            ratios.append(balmer_ratio(full_pops, 'ba'))
        
        ratio_init = balmer_ratio(pops_init, 'ba')
        ratio_final = balmer_ratio(pops_final, 'ba')
        
        ax.plot(t_dense, ratios, 'b-', linewidth=2, label='BDF integration')
        ax.axhline(ratio_final, color='red', linestyle='--', alpha=0.5, 
                   label='New steady state')
        ax.axvline(tau_slowest, color='green', linestyle=':', linewidth=2,
                   label=f'Slowest eigenvalue: τ = {tau_slowest:.2e} s')
        ax.set_xscale('log')
        ax.set_ylabel(r'H$\beta$/H$\alpha$')
        ax.set_title(f'$n_e$ = {ne:.0e} cm$^{{-3}}$')
        ax.legend(loc='lower right', fontsize=9)
    
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title(f'Step response: $kT_e$ {kTe_init} → {kTe_final} eV\n'
                      f'$n_e$ = {densities[0]:.0e} cm$^{{-3}}$')
    
    plt.tight_layout()
    plt.savefig('step_response_eigenvalues.png', dpi=150)
    plt.show()
    
    # print comparison table
    print("\n=== Eigenvalue vs BDF comparison ===")
    print(f"{'ne':>10s}  {'tau_eigenvalue':>15s}  {'note':>30s}")
    for ne in densities:
        ts = equilibration_timescales(n_levels, kTe_final, ne)
        print(f"{ne:10.0e}  {ts[-1]:15.2e}  slowest of {len(ts)} modes")

def plot_sensitivity_maps():
    import matplotlib.pyplot as plt
    
    n_levels = 15
    kTe_arr = np.logspace(0, 2.3, 80)
    ne_arr = np.logspace(10, 15, 80)
    
    print("Computing Hbeta/Halpha grid...")
    ratio_ba = compute_ratio_grid(n_levels, kTe_arr, ne_arr, "ba")
    
    print("Computing Hgamma/Hbeta grid...")
    ratio_gb = compute_ratio_grid(n_levels, kTe_arr, ne_arr, "gb")
    
    # compute derivatives in log space
    log_kTe = np.log10(kTe_arr)
    log_ne = np.log10(ne_arr)
    
    # d(ratio)/d(log10 kTe) and d(ratio)/d(log10 ne)
    # np.gradient returns derivative along each axis
    # axis=1 is kTe (columns), axis=0 is ne (rows)
    dkTe_spacing = log_kTe[1] - log_kTe[0]
    dne_spacing = log_ne[1] - log_ne[0]
    
    dba_dlogTe = np.gradient(ratio_ba, dkTe_spacing, axis=1)
    dba_dlogne = np.gradient(ratio_ba, dne_spacing, axis=0)
    
    dgb_dlogTe = np.gradient(ratio_gb, dkTe_spacing, axis=1)
    dgb_dlogne = np.gradient(ratio_gb, dne_spacing, axis=0)
    
    Te_grid, Ne_grid = np.meshgrid(kTe_arr, ne_arr)
    
    # --- Figure 1: Contour maps ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    c1 = ax1.contourf(Te_grid, Ne_grid, ratio_ba, levels=20, cmap='viridis')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$kT_e$ (eV)')
    ax1.set_ylabel(r'$n_e$ (cm$^{-3}$)')
    ax1.set_title(r'H$\beta$/H$\alpha$')
    plt.colorbar(c1, ax=ax1)
    
    c2 = ax2.contourf(Te_grid, Ne_grid, ratio_gb, levels=20, cmap='viridis')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$kT_e$ (eV)')
    ax2.set_ylabel(r'$n_e$ (cm$^{-3}$)')
    ax2.set_title(r'H$\gamma$/H$\beta$')
    plt.colorbar(c2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('contour_maps.png', dpi=150)
    plt.show()

    print("Using pcolormesh with LogNorm")
    # --- Figure 2: Sensitivity maps ---
    from matplotlib.colors import LogNorm
    
    fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # clip small values to avoid log(0)
    floor = 1e-3
    
    c1 = axes[0,0].pcolormesh(Te_grid, Ne_grid, np.abs(dba_dlogTe) + floor,
                               norm=LogNorm(), cmap='inferno', shading='auto')
    axes[0,0].set_xscale('log')
    axes[0,0].set_yscale('log')
    axes[0,0].set_xlabel(r'$kT_e$ (eV)')
    axes[0,0].set_ylabel(r'$n_e$ (cm$^{-3}$)')
    axes[0,0].set_title(r'$|\partial$(H$\beta$/H$\alpha$)$/\partial \log T_e|$')
    plt.colorbar(c1, ax=axes[0,0])
    
    c2 = axes[0,1].pcolormesh(Te_grid, Ne_grid, np.abs(dba_dlogne) + floor,
                               norm=LogNorm(), cmap='inferno', shading='auto')
    axes[0,1].set_xscale('log')
    axes[0,1].set_yscale('log')
    axes[0,1].set_xlabel(r'$kT_e$ (eV)')
    axes[0,1].set_ylabel(r'$n_e$ (cm$^{-3}$)')
    axes[0,1].set_title(r'$|\partial$(H$\beta$/H$\alpha$)$/\partial \log n_e|$')
    plt.colorbar(c2, ax=axes[0,1])
    
    c3 = axes[1,0].pcolormesh(Te_grid, Ne_grid, np.abs(dgb_dlogTe) + floor,
                               norm=LogNorm(), cmap='inferno', shading='auto')
    axes[1,0].set_xscale('log')
    axes[1,0].set_yscale('log')
    axes[1,0].set_xlabel(r'$kT_e$ (eV)')
    axes[1,0].set_ylabel(r'$n_e$ (cm$^{-3}$)')
    axes[1,0].set_title(r'$|\partial$(H$\gamma$/H$\beta$)$/\partial \log T_e|$')
    plt.colorbar(c3, ax=axes[1,0])
    
    c4 = axes[1,1].pcolormesh(Te_grid, Ne_grid, np.abs(dgb_dlogne) + floor,
                               norm=LogNorm(), cmap='inferno', shading='auto')
    axes[1,1].set_xscale('log')
    axes[1,1].set_yscale('log')
    axes[1,1].set_xlabel(r'$kT_e$ (eV)')
    axes[1,1].set_ylabel(r'$n_e$ (cm$^{-3}$)')
    axes[1,1].set_title(r'$|\partial$(H$\gamma$/H$\beta$)$/\partial \log n_e|$')
    plt.colorbar(c4, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig('sensitivity_maps.png', dpi=150)
    plt.show()
    
    # save the grid data for reuse
    np.savez('grid_data.npz', kTe_arr=kTe_arr, ne_arr=ne_arr,
             ratio_ba=ratio_ba, ratio_gb=ratio_gb)
    print("Grid data saved to grid_data.npz")

def plot_single_oscillation():
    import matplotlib.pyplot as plt
    
    n_levels = 15
    kTe_mean = 5.0
    kTe_amp = 3.0
    ne = 1e12
    freq = 1e5  # 100 kHz
    n_cycles = 5
    
    period = 1.0 / freq
    t_end = n_cycles * period
    
    pops_init = solve_steady_state(n_levels, kTe_mean, ne)
    y0 = pops_init[1:]
    
    def kTe_of_t(t):
        return kTe_mean + kTe_amp * np.sin(2 * np.pi * freq * t)
    
    def rhs(t, y):
        kTe = kTe_of_t(t)
        M, source = build_rate_matrix(n_levels, kTe, ne)
        return M @ y - source
    
    sol = solve_ivp(rhs, [0, t_end], y0, method='BDF',
                    rtol=1e-8, atol=1e-12, dense_output=True,
                    max_step=period/50)
    
    t = np.linspace(0, t_end, 2000)
    
    temps = [kTe_of_t(ti) for ti in t]
    ratios_td = []
    ratios_ss = []
    
    for ti in t:
        y = sol.sol(ti)
        full_pops = np.zeros(n_levels)
        full_pops[0] = 1.0
        full_pops[1:] = y
        ratios_td.append(balmer_ratio(full_pops, 'ba'))
        
        kTe = kTe_of_t(ti)
        pops_ss = solve_steady_state(n_levels, kTe, ne)
        ratios_ss.append(balmer_ratio(pops_ss, 'ba'))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(t * 1e6, temps, 'k-')
    ax1.set_ylabel(r'$kT_e$ (eV)')
    ax1.set_title(f'Oscillating temperature: {kTe_mean} ± {kTe_amp} eV at {freq/1e3:.0f} kHz, '
                  f'$n_e$ = {ne:.0e} cm$^{{-3}}$')
    
    ax2.plot(t * 1e6, ratios_td, 'b-', linewidth=2, label='Time-dependent')
    ax2.plot(t * 1e6, ratios_ss, 'r--', linewidth=1.5, label='Steady-state')
    ax2.set_xlabel('Time (μs)')
    ax2.set_ylabel(r'H$\beta$/H$\alpha$')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('single_oscillation.png', dpi=150)
    plt.show()

def measure_tracking(n_levels, kTe_mean, kTe_amp, ne, freq, n_cycles=5):
    """
    Returns fractional diagnostic error.
    0 = perfect agreement, larger = worse.
    """
    
    period = 1.0 / freq
    t_end = n_cycles * period
    
    pops_init = solve_steady_state(n_levels, kTe_mean, ne)
    y0 = pops_init[1:]
    
    def kTe_of_t(t):
        return kTe_mean + kTe_amp * np.sin(2 * np.pi * freq * t)
    
    def rhs(t, y):
        kTe = kTe_of_t(t)
        M, source = build_rate_matrix(n_levels, kTe, ne)
        return M @ y - source
    
    sol = solve_ivp(rhs, [0, t_end], y0, method='BDF',
                    rtol=1e-8, atol=1e-12, dense_output=True,
                    max_step=period/50)
    
    t_sample = np.linspace(3 * period, 5 * period, 200)
    
    ratios_td = []
    ratios_ss = []
    for ti in t_sample:
        y = sol.sol(ti)
        full_pops = np.zeros(n_levels)
        full_pops[0] = 1.0
        full_pops[1:] = y
        ratios_td.append(balmer_ratio(full_pops, 'ba'))
        
        kTe = kTe_of_t(ti)
        pops_ss = solve_steady_state(n_levels, kTe, ne)
        ratios_ss.append(balmer_ratio(pops_ss, 'ba'))
    
    ratios_td = np.array(ratios_td)
    ratios_ss = np.array(ratios_ss)
    
    mean_ratio = np.mean(ratios_ss)
    rms_error = np.sqrt(np.mean((ratios_td - ratios_ss)**2))
    
    return rms_error / mean_ratio

def plot_tracking_contour():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    
    n_levels = 15
    kTe_mean = 5.0
    kTe_amp = 3.0
    
    freq_arr = np.logspace(3, 8, 20)
    ne_arr = np.logspace(10, 15, 20)
    
    error = np.zeros((len(ne_arr), len(freq_arr)))
    
    for i, ne in enumerate(ne_arr):
        for j, freq in enumerate(freq_arr):
            error[i, j] = measure_tracking(n_levels, kTe_mean, kTe_amp, ne, freq)
        print(f"  ne = {ne:.0e} done")
    
    freq_grid, ne_grid = np.meshgrid(freq_arr, ne_arr)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    c = ax.pcolormesh(freq_grid, ne_grid, error, 
                       norm=LogNorm(vmin=1e-3, vmax=1e1), 
                       cmap='RdYlGn_r', shading='auto')
    ax.contour(freq_grid, ne_grid, error, levels=[0.05], 
               colors='black', linewidths=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Oscillation frequency (Hz)')
    ax.set_ylabel(r'$n_e$ (cm$^{-3}$)')
    ax.set_title(r'Steady-state diagnostic error: H$\beta$/H$\alpha$'
                 f'\n$kT_e$ = {kTe_mean} ± {kTe_amp} eV')
    plt.colorbar(c, ax=ax, label='Fractional RMS error')
    
    plt.tight_layout()
    plt.savefig('tracking_contour.png', dpi=150)
    plt.show()
    
    np.savez('tracking_data.npz', freq_arr=freq_arr, ne_arr=ne_arr, error=error)

def test_tracking():
    n_levels = 15
    for freq in [1e3, 1e5, 1e7]:
        err = measure_tracking(n_levels, 5.0, 3.0, 1e12, freq)
        print(f"freq={freq:.0e}: fractional error={err:.4f} ({err*100:.1f}%)")

def plot_vinoth_combined():
    import matplotlib.pyplot as plt
    
    n_levels = 15
    ne_arr = np.logspace(10, 15, 50)
    temps = [10, 80, 100, 200, 400]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    for kTe, color in zip(temps, colors):
        ratios_ba_no = []
        ratios_gb_no = []
        ratios_ba_yes = []
        ratios_gb_yes = []
        for ne in ne_arr:
            pops_no = solve_steady_state(n_levels, kTe, ne, use_recomb=False)
            pops_yes = solve_steady_state(n_levels, kTe, ne, use_recomb=True)
            ratios_ba_no.append(balmer_ratio(pops_no, 'ba'))
            ratios_gb_no.append(balmer_ratio(pops_no, 'gb'))
            ratios_ba_yes.append(balmer_ratio(pops_yes, 'ba'))
            ratios_gb_yes.append(balmer_ratio(pops_yes, 'gb'))
        
        ax1.plot(ne_arr, ratios_ba_no, color=color, linestyle='-', 
                 label=f'{kTe} eV (no recomb)')
        ax1.plot(ne_arr, ratios_ba_yes, color=color, linestyle='--',
                 label=f'{kTe} eV (with recomb)')
        ax2.plot(ne_arr, ratios_gb_no, color=color, linestyle='-',
                 label=f'{kTe} eV (no recomb)')
        ax2.plot(ne_arr, ratios_gb_yes, color=color, linestyle='--',
                 label=f'{kTe} eV (with recomb)')
        print(f"kTe = {kTe} eV done")
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1e10, 1e15)
    ax1.set_ylim(0.06, 0.3)
    ax1.set_xlabel(r'Electron density (cm$^{-3}$)')
    ax1.set_ylabel(r'H$\beta$/H$\alpha$')
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_title('(a)')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1e10, 1e15)
    ax2.set_ylim(0.1, 0.5)
    ax2.set_xlabel(r'Electron density (cm$^{-3}$)')
    ax2.set_ylabel(r'H$\gamma$/H$\beta$')
    ax2.legend(fontsize=7, ncol=2)
    ax2.set_title('(b)')
    
    plt.tight_layout()
    plt.savefig('vinoth_combined.png', dpi=150)
    plt.show()

def test_rate_uncertainty():
    n_levels = 15
    print("=== Balmer Ratio Uncertainty from ±20% Rate Coefficients ===")
    print(f"{'kTe (eV)':>10} {'ne (cm^-3)':>12} {'Hb/Ha low':>10} {'Hb/Ha nom':>10} {'Hb/Ha high':>10} {'spread':>8}")
    
    test_points = [
        (5.0, 1e10),   # coronal, sensitive to Te
        (5.0, 1e12),   # CR transition
        (10.0, 1e12),  # moderate
        (10.0, 1e14),  # high density
        (100.0, 1e12), # high Te
    ]
    
    for kTe, ne in test_points:
        pops_low = solve_steady_state(n_levels, kTe, ne, rate_scale=0.8)
        pops_nom = solve_steady_state(n_levels, kTe, ne, rate_scale=1.0)
        pops_high = solve_steady_state(n_levels, kTe, ne, rate_scale=1.2)
        
        r_low = balmer_ratio(pops_low, 'ba')
        r_nom = balmer_ratio(pops_nom, 'ba')
        r_high = balmer_ratio(pops_high, 'ba')
        
        spread = (r_high - r_low) / r_nom * 100
        print(f"{kTe:10.1f} {ne:12.0e} {r_low:10.4f} {r_nom:10.4f} {r_high:10.4f} {spread:7.1f}%")


if __name__ == "__main__":
    #test_oscillator_strengths()
    #test_bethe_coefficients()
    #test_excitation_rates()
    #test_ionization_and_einstein()
    #test_deexcitation()
    #test_convergence()
    #test_coronal_limit()
    #test_lte()
    #plot_vinoth_comparison()
    #plot_equilibration()
    #run_contour_plots()
    plot_sensitivity_maps()
    #run_step_response_with_eigenvalues()
    #plot_single_oscillation()
    #test_tracking()
    #plot_vinoth_combined()
    #test_rate_uncertainty()